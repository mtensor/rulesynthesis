# deepcoder eval stuff
nTest=20
salt=$1
timeout=180
for split in scan_length_original scan_simple_original scan_jump_original scan_around_right_original
	do

		python evaluate_deepcoder.py --seperate_query --fn_out_model 'dc_model_SCAN.p' --dup --batchsize 128 --timeout $timeout --n_test ${nTest} --new_test_ep $split --load_data data/SCANdata$split${salt}.p --savefile results/dcSCAN$split${salt}.p &> logs/dcSCAN$split${salt}.txt
		python evaluate_deepcoder.py --enum_only --seperate_query --fn_out_model 'dc_model_SCAN.p' --dup --batchsize 128 --timeout $timeout --n_test ${nTest} --new_test_ep $split --load_data data/SCANdata$split${salt}.p --savefile results/enumSCAN$split${salt}.p &> logs/enumSCAN$split${salt}.txt

		echo "dc :"
		grep "AVERAGE" logs/dcSCAN$split${salt}.txt
		grep "standard error" logs/dcSCAN$split${salt}.txt

		echo "enum :"
		grep "AVERAGE" logs/enumSCAN$split${salt}.txt
		grep "standard error" logs/enumSCAN$split${salt}.txt

		echo ""
		echo ""
	done

