
salt=$1
timeout=20
for split in scan_length_original scan_simple_original scan_jump_original scan_around_right_original
	do

		python scan_search.py --fn_out_model 'scan_final.p' --batchsize 128 --timeout $timeout --new_test_ep $split --dup --hack_gt --n_runs 5 --savefile results/SEARCH$split${salt}.p &>  logs/SEARCH$split${salt}.txt

		echo $split
		grep "avg" logs/SEARCH$split${salt}.txt
		echo ""

	done