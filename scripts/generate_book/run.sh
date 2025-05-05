chmod +x ./engine/katago
mkdir results

export LD_LIBRARY_PATH=LD_LIBRARY_PATH:"/root/so/"
while true
do
    chmod +x ./engine/katago
    CUDA_VISIBLE_DEVICES="4,5,6,7" ./engine/katago genbook -config book.cfg -model b18.bin.gz -html-dir results/html -book-file results/book -log-file results/log  -num-iters 10000 -save-every 5 -allow-changing-book-params

done