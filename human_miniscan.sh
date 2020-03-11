





echo sup$param.p
echo rules\_$type\_$param 
python evaluate.py --human_miniscan --seperate_query --fn_out_model 'miniscan_final.p' --batchsize 128 --timeout 30 --n_test 50 --new_test_ep rules\_$type\_$param --savefile results/ourshuman.p &> logs/ourhuman.txt
python evaluate.py --human_miniscan --seperate_query --fn_out_model 'miniscan_final.p' --nosearch --batchsize 128 --timeout 30 --new_test_ep rules\_$type\_$param --savefile results/nosearchhuman.p &> logs/nosearchhuman.txt
python train_metanet_attn.py --human_miniscan --fn_out_model 'metas2s_baseline.p' --episode_type 'rules_gen' &> logs/s2shuman.txt

echo $type
echo "ours $param :"
grep "AVERAGE" logs/ourshuman.txt
grep "standard error" logs/ourshuman.txt

echo "nosearch $param :"
grep "AVERAGE" logs/nosearchhuman.txt
grep "standard error" logs/nosearchhuman.txt

echo "seq2seq $param :"
grep "Acc Generalize (test)" logs/s2shuman.txt
grep "std error test" logs/s2shuman.txt

echo ""
echo ""