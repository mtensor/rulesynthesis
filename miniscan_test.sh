



salt=$1

type=sup
for param in 10 30 50 70
	do
		echo sup$param.p
		echo rules\_$type\_$param 
		python evaluate.py --seperate_query --fn_out_model 'miniscan_final.p' --batchsize 128 --timeout 30 --n_test 5 --new_test_ep rules\_$type\_$param --load_data data/sup$param.p --savefile results/ours$type$param${salt}.p &> logs/ours$type$param${salt}.txt
		python evaluate.py --seperate_query --fn_out_model 'miniscan_final.p' --nosearch --batchsize 128 --timeout 30 --new_test_ep rules\_$type\_$param --load_data data/sup$param.p --savefile results/nosearch$type$param${salt}.p &> logs/nosearch$type$param${salt}.txt
		python train_metanet_attn.py --fn_out_model 'metas2s_baseline.p' --episode_type 'rules_gen' --load_data data/sup$param.p &> logs/s2s$type$param${salt}.txt

		echo $type
		echo "ours $param :"
		grep "AVERAGE" logs/ours$type$param${salt}.txt
		grep "standard error" logs/ours$type$param${salt}.txt

		echo "nosearch $param :"
		grep "AVERAGE" logs/nosearch$type$param${salt}.txt
		grep "standard error" logs/nosearch$type$param${salt}.txt

		echo "seq2seq $param :"
		grep "Acc Generalize (test)" logs/s2s$type$param${salt}.txt
		grep "std error test" logs/s2s$type$param${salt}.txt

		echo ""
		echo ""
	done


type=prims
for param in 3 4 5 6
	do
		echo rules\_$type\_$param.p
		python evaluate.py --seperate_query --fn_out_model 'miniscan_final.p' --n_test 50 --batchsize 128 --timeout 30 --new_test_ep rules\_$type\_$param --load_data data/$type$param.p --savefile results/ours$type$param${salt}.p &> logs/ours$type$param${salt}.txt
		python evaluate.py --seperate_query --fn_out_model 'miniscan_final.p' --nosearch --batchsize 128 --timeout 30 --new_test_ep rules\_$type\_$param --load_data data/$type$param.p --savefile results/nosearch$type$param${salt}.p &> logs/nosearch$type$param${salt}.txt
		python train_metanet_attn.py --fn_out_model 'metas2s_baseline.p' --episode_type 'rules_gen' --load_data data/$type$param.p &> logs/s2s$type$param${salt}.txt

		echo $type
		echo "ours $param :"
		grep "AVERAGE" logs/ours$type$param${salt}.txt
		grep "standard error" logs/ours$type$param${salt}.txt

		echo "nosearch $param :"
		grep "AVERAGE" logs/nosearch$type$param${salt}.txt
		grep "standard error" logs/nosearch$type$param${salt}.txt

		echo "seq2seq $param :"
		grep "Acc Generalize (test)" logs/s2s$type$param${salt}.txt
		grep "std error test" logs/s2s$type$param${salt}.txt

		echo ""
		echo ""
	done

type=horules
for param in 3 4 5 6
	do
		echo rules\_$type\_$param.p
		python evaluate.py --seperate_query --fn_out_model 'miniscan_final.p' --batchsize 128 --timeout 30 --n_test 50 --new_test_ep rules\_$type\_$param --load_data data/$type$param.p --savefile results/ours$type$param${salt}.p &> logs/ours$type$param${salt}.txt
		python evaluate.py --seperate_query --fn_out_model 'miniscan_final.p' --nosearch --batchsize 128 --timeout 30 --new_test_ep rules\_$type\_$param --load_data data/$type$param.p --savefile results/nosearch$type$param${salt}.p &> logs/nosearch$type$param${salt}.txt
		python train_metanet_attn.py --fn_out_model 'metas2s_baseline.p' --episode_type 'rules_gen' --load_data data/$type$param.p &> logs/s2s$type$param${salt}.txt

		echo $type
		echo "ours $param :"
		grep "AVERAGE" logs/ours$type$param${salt}.txt
		grep "standard error" logs/ours$type$param${salt}.txt

		echo "nosearch $param :"
		grep "AVERAGE" logs/nosearch$type$param${salt}.txt
		grep "standard error" logs/nosearch$type$param${salt}.txt

		echo "seq2seq $param :"
		grep "Acc Generalize (test)" logs/s2s$type$param${salt}.txt
		grep "std error test" logs/s2s$type$param${salt}.txt

		echo ""
		echo ""
	done

