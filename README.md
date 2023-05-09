## Factuality metrics
Six representative factuality metrics included in the paper are as follows:

- **FactCC:** The codes and original FactCC can be downloaded from [here](https://github.com/salesforce/factCC). The four FactCCs trained with sub sampling and augmented data can be downloaded from [here](https://drive.google.com/drive/folders/1wg9jHrO90_t85ymRFBi7l6o4U7_fij_s?usp=sharing).
- **Dae:** The codes and trained model can be downloaded from [here](https://github.com/tagoyal/dae-factuality).
- **BertMnli, RobertaMnli, ElectraMnli:** The codes are included in the [baseline](./baseline), and the trained models can be downloaded [here](https://drive.google.com/drive/folders/1wg9jHrO90_t85ymRFBi7l6o4U7_fij_s?usp=sharing).
- **Feqa:** The codes and trained model can be downloaded from [here](https://github.com/esdurmus/feqa).

## Adversarial transformation
To perform adversarial transformation, please run the following commands:

```python
CUDA_VISIBLE_DEVICES=0 python ./adversarial_transformation/main.py -path DATA_PATH -save_dir SAVE_DIR -trans_type all

## CLI
Modify PATH_TO_THIS_FILE and navigate to the ROSE_NLI folder.

To generate large-scale data, use the following command, and change the last parameter to the desired number of parallel cores (if the program is interrupted, you cannot change it later, so please enter the correct number once).

If you need to rerun everything, make sure to delete all other JSON files starting with sentences in the sentences.json directory.

To resume after an interruption, simply re-enter the command.

cd /PATH_TO_THIS_FILE/ROSE_NLI

python adversarial_data_generation/main_api.py --input_prompt adversarial_data_generation/prompts/prompt_own.csv --input_sentence adversarial_data_generation/json_file/sentences.json --core_number 8

python adversarial_data_generation/main_api.py --input_prompt adversarial_data_generation/prompts/prompt_own.csv

python adversarial_data_generation/main_api.py --input_prompt adversarial_data_generation/prompts/prompt_own.csv --input_sentence adversarial_data_generation/json_file/sentences.json

