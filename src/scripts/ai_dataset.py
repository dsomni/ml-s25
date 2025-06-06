from copy import deepcopy

from datasets import Dataset
from transformers import DebertaV2Tokenizer, RobertaTokenizer


class DatasetType:
    ROBERTA = (1,)
    DEBERTA = (2,)


class AiDataset:
    def __init__(self, model_name, model_type):
        self.tokenizer = None
        if model_type == DatasetType.ROBERTA:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        elif model_type == DatasetType.DEBERTA:
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        else:
            raise NotImplementedError

    def tokenize_function(self, examples, padding=False):
        tz = self.tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=128,  # 1024,
            add_special_tokens=True,
            return_token_type_ids=False,
        )

        return tz

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def get_dataset(self, df):
        """
        Main api for creating the Science Exam dataset
        :param df: input dataframe
        :type df: pd.DataFrame
        :return: the created dataset
        :rtype: Dataset
        """
        df = deepcopy(df)
        task_dataset = Dataset.from_pandas(df)

        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)

        return task_dataset
