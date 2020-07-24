# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 14:34
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com

# model name map to run dir
import os
import json
import shutil
from functional import seq
import pickle
import collections
import numpy as np
import argparse

MODEL_MAP = {
    "args_train_models_1_electra_large_32_480_5e-05_2_1": "electra_master",
    "args_train_models_2_electra_large_32_384_5e-05_2_2": "electra_atrlp",
    "args_train_models_2_electra_large_32_480_5e-05_2_2": "electra_atrlp",
    'atrlp_models_1': "electra_atrlp",
    'atrlp_models_9': "electra_atrlp",
    'lr_epoch_models_3.0000000000000004e-05_2_3': "electra_atrlp",
    'lr_epoch_models_6e-05_2_1': "electra_atrlp",
    'lr_epoch_models_6e-05_3_1': "electra_atrlp",
    'albert_args_train_models_2_albert_xxlarge_v1_32_384_2e-05_2_0': "ALBERT_master",
    'albert_args_train_models_2_albert_xxlarge_v2_32_384_2e-05_2_0': "ALBERT_master",
    'albert_args_train_models_3_albert_xlarge_v2_32_384_2e-05_2_0': "ALBERT_master",
    'albert_args_train_models_3_albert_xxlarge_v1_32_384_2e-05_2_0': "ALBERT_master",
    'albert_args_train_models_3_albert_xxlarge_v2_32_384_2e-05_2_0': "ALBERT_master",
    'albert_args_train_answer_models_1_albert_xlarge_v1_32_384_2e-05_2_0': "ALBERT_answer_model",
    'albert_args_train_answer_models_2_albert_xxlarge_v2_32_384_2e-05_2_0': "ALBERT_answer_model",
    'albert_args_train_answer_models_3_albert_xxlarge_v2_32_384_2e-05_2_0': "ALBERT_answer_model",
    'args_train_pv_models_3_electra_large_24_480_3e-05_2_0': "electra_pv",
    'args_train_pv_models_2_electra_large_32_512_5e-05_2_0': "electra_pv",
}


def eval_a_model(model_dir, model_name, model_type, max_seq_len, predict_batch_size, tpu_address):
    run_dir = MODEL_MAP[model_name]
    if model_type == "albert":
        config_file = f"gs://squad_cx/albert_data/pretrain_models/{model_name}/albert_config.json"
        output_dir = f"results/{model_name}"
        predict_file = f"gs://squad_cx/albert_data/inputs/dev.json"
        predict_feature_file = f"gs://squad_cx/albert_data/features/predict_features_{max_seq_len}_128_64"
        predict_feature_left_file = f"gs://squad_cx/albert_data/features/predict_features_left_{max_seq_len}_128_64"
        init_checkpoint = f"gs://squad_cx/albert_data/pretrain_models/{model_name}/model.ckpt-best"
        spm_model_file = f"gs://squad_cx/albert_data/pretrain_models/{model_name}/30k-clean.model"
        xargs = f"gsutil cp {spm_model_file} ./"
        os.system(xargs)

        xargs = f""" cd {run_dir} && \
                    python3 run_squad_v2.py \
                      --albert_config_file={config_file} \
                      --output_dir={output_dir} \
                      --predict_file={predict_file} \
                      --predict_feature_file={predict_feature_file} \
                      --predict_feature_left_file={predict_feature_left_file} \
                      --init_checkpoint={init_checkpoint} \
                      --spm_model_file=30k-clean.model \
                      --max_seq_length={max_seq_len} \
                      --do_train=False \
                      --do_predict=True \
                      --predict_batch_size={predict_batch_size} \
                      --save_checkpoints_steps=100000 \
                      --n_best_size=20 \
                      --use_tpu=True \
                      --tpu_name={tpu_address}
                    """
        os.system(xargs)

        if os.path.exists(os.path.join(output_dir, "predictions.json")):
            shutil.move(os.path.join(output_dir, "predictions.json"), os.path.join(output_dir, "squad_preds.json"))
        if os.path.exists(os.path.join(output_dir, "null_odds.json")):
            shutil.move(os.path.join(output_dir, "null_odds.json"), os.path.join(output_dir, "squad_null_odds.json"))

    elif model_type == "electra":
        xargs = f"gsutil -m cp -r {model_dir} gs://squad_cx/electra_data/models/{model_name}"
        os.system(xargs)

        xargs = f"""cd {run_dir} && python run_finetuning.py   --data-dir=gs://squad_cx/electra_data --model-name={model_name}   --hparams '{{"model_size": "large", "task_names": ["squad"], "num_train_epochs": 2, "use_tpu": true, "num_tpu_cores": 8, "tpu_name": "{tpu_address}", "train_batch_size": 32, "eval_batch_size": {predict_batch_size}, "predict_batch_size": {predict_batch_size}, "max_seq_length": {max_seq_len}, "use_tfrecords_if_existing": false, "num_trials": 1, "do_train": false, "do_eval": true, "save_checkpoints_steps": 100000 }}' """
        os.system(xargs)

        xargs = f"gsutil -m cp -r gs://squad_cx/electra_data/models/{model_name}/results/squad_qa ./results/{model_name}"
        os.system(xargs)
    else:
        raise


def stage1_qa_bagging(input_file):
    results_dir = "results"
    models = os.listdir(results_dir)
    assert len(models) == 13
    all_nbest = []
    all_odds = []
    all_preds = []
    for dire in [os.path.join(results_dir, d) for d in models]:
        all_nbest.append(pickle.load(open(os.path.join(dire, 'eval_all_nbest.pkl'), 'rb')))
        all_odds.append(json.load(open(os.path.join(dire, 'squad_null_odds.json'), 'r', encoding='utf-8')))
        all_preds.append(json.load(open(os.path.join(dire, 'squad_preds.json'), 'r', encoding='utf-8')))
    qids = seq(all_preds[0].keys()).list()

    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    for qid in qids:
        bagging_preds[qid] = (seq([nbest[qid][0] for nbest in all_nbest])
                              .sorted(key=lambda x: x['probability'])
                              ).list()[-1]['text']
        bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

    shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    output_bagging_preds_file = os.path.join(results_dir, "stage1_qa_bagging_preds.json")
    output_bagging_odds_file = os.path.join(results_dir, "stage1_qa_bagging_odds.json")
    output_bagging_eval_file = os.path.join(results_dir, "stage1_qa_bagging_eval.json")

    json.dump(bagging_preds, open(output_bagging_preds_file, 'w', encoding='utf-8'))
    json.dump(bagging_odds, open(output_bagging_odds_file, 'w', encoding='utf-8'))

    xargs = f"python eval.py {input_file} {output_bagging_preds_file} --na-prob-file {output_bagging_odds_file} -o {output_bagging_eval_file}"
    os.system(xargs)


def build_pv_data(input_file):
    results_dir = "results"
    stage1_qa_bagging_preds_file = os.path.join(results_dir, "stage1_qa_bagging_preds.json")
    output_pv_data_file = os.path.join(results_dir, "pv_data_file.json")
    dev = json.load(open(input_file))
    preds = json.load(open(stage1_qa_bagging_preds_file))

    for article in dev['data']:
        for paragraph in article["paragraphs"]:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qid = qa['id']
                pred = preds[qid]
                qa['is_impossible'] = True
                qa['plausible_answers'] = [{'text': pred, 'answer_start': 1}]

    json.dump(dev, open(output_pv_data_file, 'w', encoding='utf-8'))
    print("generate pv data finished !")
    xargs = f"gsutil cp {output_pv_data_file} gs://squad_cx/electra_large/finetuning_data/squad/dev.json"
    os.system(xargs)

    print("update electra pv data !")
    shutil.rmtree(results_dir)
    os.makedirs(results_dir)


def stage2_answer_verifier_step_one(input_file):
    results_dir = "results"
    models = os.listdir(results_dir)
    assert len(models) == 4
    all_odds = []
    for dire in [os.path.join(results_dir, d) for d in models]:
        if "albert" in dire:
            all_odds.append(json.load(
                open(os.path.join(dire, 'squad_preds.json'), 'r', encoding='utf-8')))
        else:
            all_odds.append(json.load(
                open(os.path.join(dire, 'squad_null_odds.json'), 'r', encoding='utf-8')))

    stage1_bagging_preds_file = os.path.join(results_dir, "stage1_qa_bagging_preds.json")
    stage1_bagging_odds_file = os.path.join(results_dir, "stage1_qa_bagging_odds.json")
    stage1_bagging_eval_file = os.path.join(results_dir, "stage1_qa_bagging_eval.json")

    stage1_qa_bagging_preds = json.load(open(stage1_bagging_preds_file, 'r', encoding='utf-8'))
    stage1_qa_bagging_odds = json.load(open(stage1_bagging_odds_file, 'r', encoding='utf-8'))
    stage1_qa_bagging_eval = json.load(open(stage1_bagging_eval_file, 'r', encoding='utf-8'))

    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    for qid in stage1_qa_bagging_preds:
        bagging_preds[qid] = stage1_qa_bagging_preds[qid]
        if stage1_qa_bagging_odds[qid] > stage1_qa_bagging_eval['best_exact_thresh']:
            bagging_preds[qid] = ""
        bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

    shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    output_bagging_preds_file = os.path.join(results_dir, "stage2_verifier_bagging_preds.json")
    output_bagging_odds_file = os.path.join(results_dir, "stage2_verifier_bagging_odds.json")
    output_bagging_eval_file = os.path.join(results_dir, "stage2_verifier_bagging_eval.json")

    json.dump(bagging_preds, open(output_bagging_preds_file, 'w', encoding='utf-8'))
    json.dump(bagging_odds, open(output_bagging_odds_file, 'w', encoding='utf-8'))

    xargs = f"python eval.py {input_file} {output_bagging_preds_file} --na-prob-file {output_bagging_odds_file} -o {output_bagging_eval_file}"
    os.system(xargs)


def stage2_answer_verifier_step_two(input_file):
    results_dir = "results"
    models = os.listdir(results_dir)
    assert len(models) == 1
    all_odds = []
    for dire in [os.path.join(results_dir, d) for d in models]:
        if "albert" in dire:
            all_odds.append(json.load(
                open(os.path.join(dire, 'squad_preds.json'), 'r', encoding='utf-8')))
        else:
            all_odds.append(json.load(
                open(os.path.join(dire, 'squad_null_odds.json'), 'r', encoding='utf-8')))
    stage2_step_one_bagging_preds_file = os.path.join(results_dir, "stage2_verifier_bagging_preds.json")
    stage2_step_one_bagging_odds_file = os.path.join(results_dir, "stage2_verifier_bagging_odds.json")
    stage2_step_one_bagging_eval_file = os.path.join(results_dir, "stage2_verifier_bagging_eval.json")

    stage2_step_one_bagging_preds = json.load(open(stage2_step_one_bagging_preds_file, 'r', encoding='utf-8'))
    stage2_step_one_bagging_odds = json.load(open(stage2_step_one_bagging_odds_file, 'r', encoding='utf-8'))
    stage2_step_one_bagging_eval = json.load(open(stage2_step_one_bagging_eval_file, 'r', encoding='utf-8'))

    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    for qid in stage2_step_one_bagging_preds:
        bagging_preds[qid] = stage2_step_one_bagging_preds[qid]
        if stage2_step_one_bagging_odds[qid] > stage2_step_one_bagging_eval['best_exact_thresh']:
            bagging_preds[qid] = ""
        bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

    shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    output_bagging_preds_file = os.path.join(results_dir, "stage2_verifier_bagging_preds.json")
    output_bagging_odds_file = os.path.join(results_dir, "stage2_verifier_bagging_odds.json")
    output_bagging_eval_file = os.path.join(results_dir, "stage2_verifier_bagging_eval.json")

    json.dump(bagging_preds, open(output_bagging_preds_file, 'w', encoding='utf-8'))
    json.dump(bagging_odds, open(output_bagging_odds_file, 'w', encoding='utf-8'))

    xargs = f"python eval.py {input_file} {output_bagging_preds_file} --na-prob-file {output_bagging_odds_file} -o {output_bagging_eval_file}"
    os.system(xargs)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--input-file', required=True, help="eval file")
    parser.add_argument('--tpu-address', required=True, help="eval file")
    args = parser.parse_args()
    tpu_address = args.tpu_address

    xargs = f"gsutil cp {args.input_file} gs://squad_cx/albert_data/inputs/dev.json"
    os.system(xargs)

    xargs = f"gsutil cp {args.input_file} gs://squad_cx/electra_data/finetuning_data/squad/dev.json"
    os.system(xargs)

    predict_batch_size = 32
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/1_electra_large_32_480_5e-05_2_1",
                 "args_train_models_1_electra_large_32_480_5e-05_2_1", "electra", 480, predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/2_electra_large_32_384_5e-05_2_2",
                 "args_train_models_2_electra_large_32_384_5e-05_2_2", "electra", 384, predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/2_electra_large_32_480_5e-05_2_2",
                 "args_train_models_2_electra_large_32_480_5e-05_2_2", "electra", 480, predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/squad_model_1", "atrlp_models_1", "electra", 512,
                 predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/squad_model_9", "atrlp_models_9", "electra", 512,
                 predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/3.0000000000000004e-05_2_3",
                 "lr_epoch_models_3.0000000000000004e-05_2_3", "electra", 512, predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/6e-05_2_1", "lr_epoch_models_6e-05_2_1", "electra", 512,
                 predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/6e-05_3_1", "lr_epoch_models_6e-05_3_1", "electra", 512,
                 predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/2_albert_xxlarge_v1_32_384_2e-05_2_0",
                 "albert_args_train_models_2_albert_xxlarge_v1_32_384_2e-05_2_0", "albert", 384, predict_batch_size,
                 tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/2_albert_xxlarge_v2_32_384_2e-05_2_0",
                 "albert_args_train_models_2_albert_xxlarge_v2_32_384_2e-05_2_0", "albert", 384, predict_batch_size,
                 tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/3_albert_xlarge_v2_32_384_2e-05_2_0",
                 "albert_args_train_models_3_albert_xlarge_v2_32_384_2e-05_2_0", "albert", 384, predict_batch_size,
                 tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/3_albert_xxlarge_v1_32_384_2e-05_2_0",
                 "albert_args_train_models_3_albert_xxlarge_v1_32_384_2e-05_2_0", "albert", 384, predict_batch_size,
                 tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/qa_models/3_albert_xxlarge_v2_32_384_2e-05_2_0",
                 "albert_args_train_models_3_albert_xxlarge_v2_32_384_2e-05_2_0", "albert", 384, predict_batch_size,
                 tpu_address)

    stage1_qa_bagging(args.input_file)
    build_pv_data(args.input_file)

    eval_a_model("gs://squad_cx/my_ensemble_models/answer_verifier_models/1_albert_xlarge_v1_32_384_2e-05_2_0",
                 "albert_args_train_answer_models_1_albert_xlarge_v1_32_384_2e-05_2_0", "albert", 384,
                 predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/answer_verifier_models/2_albert_xxlarge_v2_32_384_2e-05_2_0",
                 "albert_args_train_answer_models_2_albert_xxlarge_v2_32_384_2e-05_2_0", "albert", 384,
                 predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/answer_verifier_models/3_albert_xxlarge_v2_32_384_2e-05_2_0",
                 "albert_args_train_answer_models_3_albert_xxlarge_v2_32_384_2e-05_2_0", "albert", 384,
                 predict_batch_size, tpu_address)
    eval_a_model("gs://squad_cx/my_ensemble_models/answer_verifier_models/3_electra_large_24_480_3e-05_2_0",
                 "args_train_pv_models_3_electra_large_24_480_3e-05_2_0", "electra", 480, predict_batch_size,
                 tpu_address)

    stage2_answer_verifier_step_one(args.input_file)

    eval_a_model("gs://squad_cx/my_ensemble_models/answer_verifier_models/2_electra_large_32_512_5e-05_2_0",
                 "args_train_pv_models_2_electra_large_32_512_5e-05_2_0", "electra", 512, predict_batch_size,
                 tpu_address)

    stage2_answer_verifier_step_two(args.input_file)

    xargs = f"gsutil -m cp -r results gs://squad_cx"
    os.system(xargs)


if __name__ == '__main__':
    main()
