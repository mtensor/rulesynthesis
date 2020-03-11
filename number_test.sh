
timeout=45
salt=$1
for lang in en es zh ja it el ko fr vi
	do
		python evaluate.py --seperate_query --fn_out_model WordToNum.p --type WordToNumber --batchsize 128 --new_test_ep lang_$lang --load_data data/$lang${salt}.p --n_test 5 --dup --timeout $timeout --savefile results/ours$lang${salt}.p &> logs/ours$lang${salt}.txt
		python evaluate.py --seperate_query --fn_out_model WordToNum.p --type WordToNumber --batchsize 128 --load_data data/$lang${salt}.p --dup --savefile results/nosearch$lang${salt}.p --nosearch &> logs/nosearch$lang${salt}.txt
		python train_metanet_attn.py --fn_out_model MetaNetw2num.p --episode_type 'wordToNumber' --load_data data/$lang${salt}.p --dup &> logs/s2s$lang${salt}.txt

		echo $lang
		echo "ours:"
		grep "AVERAGE" logs/ours$lang${salt}.txt
		grep "standard error" logs/ours$lang${salt}.txt

		echo "nosearch:"
		grep "AVERAGE" logs/nosearch$lang${salt}.txt
		grep "standard error" logs/nosearch$lang${salt}.txt

		echo "seq2seq:"
		grep "Acc Generalize (test)" logs/s2s$lang${salt}.txt
		grep "std error test" logs/s2s$lang${salt}.txt

		echo ""
		echo ""
	done 