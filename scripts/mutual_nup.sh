python run_MUTUAL.py \
--dialog_electra_weights weights/NUP_dialog_electra_2.pt \
--dialog_modeling_weights weights/NUP_dialog_modeling_2.pt \
--max_seq_length 256 \
--train_batch_size 6 \
--eval_batch_size 2 \
--learning_rate 4e-6 \
--num_train_epochs 6 \
--gradient_accumulation_steps 3 \
--seed 123456