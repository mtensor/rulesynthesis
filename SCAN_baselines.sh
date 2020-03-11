
salt=$1
timeout=180
for split in scan_length_original scan_simple_original scan_jump_original scan_around_right_original
	do

		python evaluate.py --fn_out_model 'scan_final.p' --batchsize 128 --timeout $timeout --dup --hack_gt --nosearch --new_test_ep $split --load_data data/SCANdata$split${salt}.p --savefile results/nosearch$split${salt}.p &>  logs/nosearch$split${salt}.txt
		python train_metanet_attn.py --num_episodes 1000000 --fn_out_model 'scan_metas2s_baseline.p' --episode_type 'scan_random' --new_test_ep $split --load_data data/SCANdata$split${salt}.p --dup &> logs/s2s$split${salt}.txt
		python MCMC_baseline.py --mode MCMC --num_traces $(($timeout * 60)) --load_data data/SCANdata$split${salt}.p --savefile results/MCMC$split${salt}.p &> logs/MCMC$split${salt}.txt
		python MCMC_baseline.py --mode sample --timeout $timeout --load_data data/SCANdata$split${salt}.p --savefile results/sample$split${salt}.p &> logs/sample$split${salt}.txt

		echo "nosearch:"
		grep "AVERAGE" logs/nosearch$split${salt}.txt
		grep "standard error" logs/nosearch$split${salt}.txt
		echo ""

		echo "seq2seq:"
		grep "Acc Generalize (test)" logs/s2s$split${salt}.txt
		grep "std error test" logs/s2s$split${salt}.txt

		echo "MCMC:"
		grep "AVERAGE" logs/MCMC$split${salt}.txt
		grep "standard error" logs/MCMC$split${salt}.txt

		echo "sample from prior:"
		grep "AVERAGE" logs/sample$split${salt}.txt
		grep "standard error" logs/sample$split${salt}.txt


		echo ""
		echo ""
	done
