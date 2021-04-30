for device in "CPU" "GPU"
do 
	for transf in "no" "yes"
	do
		echo --- ${device}-${transf} ---
		MODEL1=repro_results_${device}_${transf}-transformer/exp_1/
		MODEL2=repro_results_${device}_${transf}-transformer/exp_2/	
		diff -qr $MODEL1 $MODEL2 
		if [ ${transf} = "yes" ]; then
			echo -n "model-best: "
			python compare_torch.py ${MODEL1}/model-best/transformer/model/pytorch_model.bin ${MODEL2}/model-best/transformer/model/pytorch_model.bin
			echo -n "model-last: "
			python compare_torch.py ${MODEL1}/model-last/transformer/model/pytorch_model.bin ${MODEL2}/model-last/transformer/model/pytorch_model.bin
		fi
		echo " "
	done
done

