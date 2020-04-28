#!/bin/sh


# to train a network
# ident: num id of model
# test_model: dir to save model

python3 --ident 20 --test_model saved_models --model_name AudioToJoints --device cpu --seq_len 1 --batch_size 10 --hidden_size 100

# to test a network (on audio files only rn)
python3 --ident 20 --test_model saved_models --model_name AudioToJoints --device cpu --seq_len 1 --batch_size 10 --hidden_size 100 --audio_file link_to_audio
