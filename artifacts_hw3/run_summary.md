| run_name | command | model_name_or_path | test_before_f1 | test_after_f1 | delta_f1 | test_after_precision | test_after_recall | eval_loss | epoch | eval_runtime | eval_samples_per_second | eval_steps_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ner_from_entity_mlm_with_synthetic_titles_10k | train-ner | artifacts_hw3/mlm_rubert_tiny2_entity_e1/checkpoint-best | 0.0326 | 0.8328 | 0.8002 | 0.7972 | 0.8717 |  |  |  |  |  |
| ner_with_synthetic_titles_10k | train-ner | cointegrated/rubert-tiny2 | 0.0318 | 0.8251 | 0.7933 | 0.7856 | 0.8688 |  |  |  |  |  |
| ner_from_mlm_rubert_tiny2_entity_e1 | train-ner | artifacts_hw3/mlm_rubert_tiny2_entity_e1/checkpoint-best | 0.0326 | 0.7721 | 0.7394 | 0.7372 | 0.8105 |  |  |  |  |  |
| ner_from_mlm_rubert_tiny2_whole_word_e1 | train-ner | artifacts_hw3/mlm_rubert_tiny2_whole_word_e1/checkpoint-best | 0.0322 | 0.7487 | 0.7165 | 0.7102 | 0.7916 |  |  |  |  |  |
| baseline_rubert_tiny2_e1 | train-ner | cointegrated/rubert-tiny2 | 0.0318 | 0.7367 | 0.7050 | 0.6953 | 0.7834 |  |  |  |  |  |
| smoke_ner_tiny_v2 | train-ner | hf-internal-testing/tiny-random-bert | 0.0414 | 0.0420 | 0.0007 | 0.0244 | 0.1519 |  |  |  |  |  |
| mlm_rubert_tiny2_entity_e1 | train-mlm | cointegrated/rubert-tiny2 |  |  |  |  |  | 4.6614 | 1.0000 | 53.5351 | 48.2300 | 1.5130 |
| mlm_rubert_tiny2_whole_word_e1 | train-mlm | cointegrated/rubert-tiny2 |  |  |  |  |  | 3.7948 | 1.0000 | 41.1995 | 62.6710 | 1.9660 |
| smoke_mlm_entity | train-mlm | hf-internal-testing/tiny-random-bert |  |  |  |  |  | 7.0099 | 1.0000 | 0.0698 | 458.1700 | 57.2710 |
| smoke_mlm_whole_word | train-mlm | hf-internal-testing/tiny-random-bert |  |  |  |  |  | 7.0203 | 1.0000 | 0.0769 | 416.0060 | 52.0010 |