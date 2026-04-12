#!/usr/bin/env python3

import argparse
from pathlib import Path

import onnx
import numpy as np
from onnx import TensorProto, checker, helper


SPATIAL_FEATURES_BY_VERSION = {
    8: 22,
    9: 22,
    10: 22,
    11: 22,
}

GLOBAL_FEATURES_BY_VERSION = {
    8: 19,
    9: 19,
    10: 19,
    11: 19,
}


def make_shape_nodes(base_name, batch_dim_name, second_dim):
    second_dim_name = f"{base_name}_second_dim"
    shape_name = f"{base_name}_shape"
    output_name = base_name
    return [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[second_dim_name],
            value=helper.make_tensor(
                name=f"{second_dim_name}_value",
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[second_dim],
            ),
        ),
        helper.make_node(
            "Concat",
            inputs=[batch_dim_name, second_dim_name],
            outputs=[shape_name],
            axis=0,
        ),
        helper.make_node(
            "ConstantOfShape",
            inputs=[shape_name],
            outputs=[output_name],
            value=helper.make_tensor(
                name=f"{base_name}_zero",
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=[0.0],
            ),
        ),
    ]


def build_model(name, model_version, nn_len, has_mask):
    spatial_features = SPATIAL_FEATURES_BY_VERSION[model_version]
    global_features = GLOBAL_FEATURES_BY_VERSION[model_version]
    policy_size = 4 * (nn_len + 1)

    input_spatial = helper.make_tensor_value_info(
        "input_spatial",
        TensorProto.FLOAT,
        ["batch", spatial_features, nn_len],
    )
    input_global = helper.make_tensor_value_info(
        "input_global",
        TensorProto.FLOAT,
        ["batch", global_features],
    )

    out_policy = helper.make_tensor_value_info(
        "out_policy",
        TensorProto.FLOAT,
        ["batch", policy_size],
    )
    out_value = helper.make_tensor_value_info(
        "out_value",
        TensorProto.FLOAT,
        ["batch", 3],
    )
    out_miscvalue = helper.make_tensor_value_info(
        "out_miscvalue",
        TensorProto.FLOAT,
        ["batch", 10],
    )
    out_moremiscvalue = helper.make_tensor_value_info(
        "out_moremiscvalue",
        TensorProto.FLOAT,
        ["batch", 8],
    )
    out_ownership = helper.make_tensor_value_info(
        "out_ownership",
        TensorProto.FLOAT,
        ["batch", nn_len],
    )

    nodes = [
        helper.make_node("Shape", inputs=["input_global"], outputs=["input_global_shape"]),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["batch_index"],
            value=helper.make_tensor(
                name="batch_index_value",
                data_type=TensorProto.INT64,
                dims=[],
                vals=[0],
            ),
        ),
        helper.make_node(
            "Gather",
            inputs=["input_global_shape", "batch_index"],
            outputs=["batch_dim_scalar"],
            axis=0,
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=["batch_dim_scalar", "unsqueeze_axes"],
            outputs=["batch_dim"],
        ),
    ]
    nodes.extend(make_shape_nodes("out_policy", "batch_dim", policy_size))
    nodes.extend(make_shape_nodes("out_value", "batch_dim", 3))
    nodes.extend(make_shape_nodes("out_miscvalue", "batch_dim", 10))
    nodes.extend(make_shape_nodes("out_moremiscvalue", "batch_dim", 8))
    nodes.extend(make_shape_nodes("out_ownership", "batch_dim", nn_len))

    graph = helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=[input_spatial, input_global],
        outputs=[
            out_policy,
            out_value,
            out_miscvalue,
            out_moremiscvalue,
            out_ownership,
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="katago-dummy-onnx-generator",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    unsqueeze_axes_init = helper.make_tensor(
        name="unsqueeze_axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0],
    )
    model.graph.initializer.extend([unsqueeze_axes_init])
    helper.set_model_props(
        model,
        {
            "modelVersion": str(model_version),
            "name": name,
            "num_spatial_inputs": str(spatial_features),
            "num_global_inputs": str(global_features),
            "has_mask": "true" if has_mask else "false",
            "pos_len_x": str(nn_len),
            "pos_len_y": "1",
            "is_qat": "false",
            "is_simplified": "false",
            "is_int8": "false",
            "model_config": '{"dummy":true}',
        },
    )
    checker.check_model(model)
    return model


def run_self_test(model_path, model_version, nn_len, batch_size):
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("self-test failed: onnxruntime is required") from exc

    spatial_features = SPATIAL_FEATURES_BY_VERSION[model_version]
    global_features = GLOBAL_FEATURES_BY_VERSION[model_version]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    spatial = np.zeros((batch_size, spatial_features, nn_len), dtype=np.float32)
    global_in = np.zeros((batch_size, global_features), dtype=np.float32)
    outputs = session.run(
        ["out_policy", "out_value", "out_miscvalue", "out_moremiscvalue", "out_ownership"],
        {"input_spatial": spatial, "input_global": global_in},
    )
    expected_shapes = [
        (batch_size, 4 * (nn_len + 1)),
        (batch_size, 3),
        (batch_size, 10),
        (batch_size, 8),
        (batch_size, nn_len),
    ]
    for out, shape in zip(outputs, expected_shapes):
        if out.shape != shape:
            raise RuntimeError(f"self-test shape mismatch: got {out.shape}, expected {shape}")
        if not np.allclose(out, 0.0):
            raise RuntimeError("self-test value mismatch: output is not all zeros")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--len", type=int, required=True, dest="nn_len")
    parser.add_argument("--model-version", type=int, default=11, choices=sorted(SPATIAL_FEATURES_BY_VERSION))
    parser.add_argument("--name", default="dummy_zero_model")
    parser.add_argument("--has-mask", action="store_true")
    parser.add_argument("--self-test-batch-size", type=int, default=2)
    parser.add_argument("--skip-self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nn_len <= 0:
        raise SystemExit("len must be positive")

    model = build_model(
        name=args.name,
        model_version=args.model_version,
        nn_len=args.nn_len,
        has_mask=args.has_mask,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    if not args.skip_self_test:
        run_self_test(args.output, args.model_version, args.nn_len, args.self_test_batch_size)
    print(args.output)


if __name__ == "__main__":
    main()
