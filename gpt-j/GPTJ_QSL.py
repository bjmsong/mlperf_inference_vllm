import mlperf_loadgen as lg
from dataset import Dataset

class GPTJ_QSL():
    def __init__(self, dataset_path: str, max_examples: int, model_path: str):
        self.dataset_path = dataset_path
        self.max_examples = max_examples
        self.model_path = model_path

        # creating data object for QSL
        self.data_object = Dataset(
                self.dataset_path, total_count_override=self.max_examples, model_path=self.model_path)
        
        # construct QSL from python binding
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        print("Finished constructing QSL.")

def get_GPTJ_QSL(dataset_path: str, max_examples: int, model_path: str):
    return GPTJ_QSL(dataset_path, max_examples, model_path)