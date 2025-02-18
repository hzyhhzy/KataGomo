#!/bin/bash -eu

POSLEN=13
LRSCALE=0.5
SAMPLEPEREPOCH=1000000

set -o pipefail
{
# Runs training in $BASEDIR/train/$TRAININGNAME
# Should be run once per persistent training process.
# Outputs results in torchmodels_toexport/ in an ongoing basis (EXPORTMODE == "main").
# Or, to torchmodels_toexport_extra/ (EXPORTMODE == "extra").
# Or just trains without exporting (EXPORTMODE == "trainonly").

if [[ $# -lt 5 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME MODELKIND BATCHSIZE EXPORTMODE OTHERARGS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "DATADIR containing shuffled selfplay data"
    echo "TRAININGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train, like b10c128, see ../modelconfigs.py"
    echo "BATCHSIZE number of samples to concat together per batch for training, must match shuffle"
    echo "EXPORTMODE 'main': train and export for selfplay. 'extra': train and export extra non-selfplay model. 'trainonly': train without export"
    exit 0
fi
BASEDIR="$1"
shift
DATADIR="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift
BATCHSIZE="$1"
shift
EXPORTMODE="$1"
shift

#------------------------------------------------------------------------------
set -x

mkdir -p "$BASEDIR"/train/"$TRAININGNAME"

if [ "$EXPORTMODE" == "main" ]
then
    EXPORT_SUBDIR=torchmodels_toexport
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "extra" ]
then
    EXPORT_SUBDIR=torchmodels_toexport_extra
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "trainonly" ]
then
    EXPORT_SUBDIR=torchmodels_toexport_extra
    EXTRAFLAG="-no-export"
else
    echo "EXPORTMODE was not 'main' or 'extra' or 'trainonly', run with no arguments for usage"
    exit 1
fi

     #-intermediate-loss-scale 0.0 \
     #-main-loss-scale 1.0 \
time python ./train.py \
     -traindir "$BASEDIR"/train/"$TRAININGNAME" \
     -datadir "$DATADIR" \
     -exportdir "$BASEDIR"/"$EXPORT_SUBDIR" \
     -exportprefix "$TRAININGNAME" \
     -max-epochs-this-instance 1 \
     -pos-len "$POSLEN" \
     -samples-per-epoch "$SAMPLEPEREPOCH" \
     -lr-scale "$LRSCALE" \
     -use-fp16 \
     -swa-scale 16.0 \
     -swa-period-samples 100000 \
     -gnorm-clip-scale 4.0 \
     -batch-size "$BATCHSIZE" \
     -model-kind "$MODELKIND" \
     $EXTRAFLAG \
     "$@" \
     2>&1 | tee -a "$BASEDIR"/train/"$TRAININGNAME"/stdout.txt

exit 0
}
