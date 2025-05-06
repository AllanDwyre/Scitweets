import spacy
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import os


def load_nlp_model() -> spacy.Language:
	try:
		return spacy.load('en_core_web_sm')
	except OSError:
		spacy.cli.download('en_core_web_sm')
		return spacy.load('en_core_web_sm')
	
def print_color(text: str, type: str) -> None:
	types = {
		"info":    ("\033[94m", "ℹ️ "),
		"error":   ("\033[91m", "❌ "),
		"debug":   ("\033[90m", "🐞 "),
		"warning": ("\033[93m", "⚠️ "),
		"success": ("\033[92m", "✅ "),	
	}

	reset = "\033[0m"
	start_color, icon = types.get(type, ("", ""))
	print(icon, f"{start_color}{text}{reset}")

def parse_ngram_range(ngram_str: str) -> Tuple[int, int]:
	"""Converts a string like "(1,2)" to a tuple (1, 2)"""
	return tuple(map(int, ngram_str.strip("() ").split(",")))


@dataclass
class VectorizerConfig:
	ngrams: Optional[Tuple[int, int]] = (1,1)
	max_features: int = None

@dataclass
class Model:
	name		: str
	data		: Dict[str, Any]

	def get_config(self, config_name):
		return self.data.get(config_name)
		
	def get_vectorizer(self):
		vectorizer : Dict[str, Any] = self.data.get("vectorizer")
		vectorizer_type : Dict[str, Any] = self.data.get("vectorizer_type")
		
		if not vectorizer:
			print_color("No vectorizer found", "warning")
			return None
		
		vectorizer_type : Dict[str, Any] = vectorizer.get("vectorizer_type")
		if not vectorizer_type:
			print_color("vectorizer : type not found", "warning")

		ngrams = parse_ngram_range(vectorizer.get("ngram_range", "(1,2)"))
		max_features = vectorizer.get("max_features", 1000)

		if(vectorizer_type == "bow"):
			return CountVectorizer(ngram_range = ngrams, max_features = max_features)
		return TfidfVectorizer(ngram_range = ngrams, max_features = max_features)
	
	def get_vectorizer_name(self) -> str:
		vectorizer : Dict[str, Any] = self.data.get("vectorizer")
		if not vectorizer:
			print_color("No vectorizer found", "warning")
			return "None"
		return vectorizer.get("vectorizer_type", "None")
	
@dataclass
class StepConfig:
	config_path	: str
	step_name	: str
	result_dir	: str
	static_dir	: str
	random_state: int
	test_ratio	: float
	tf_idf		: VectorizerConfig
	bow			: VectorizerConfig
	saved_data	: dict[str, dict] = field(default_factory=lambda: defaultdict(dict))

	@property
	def data(self) -> Dict[str, Any]:
		with open(self.config_path, 'r') as f:
			full_config = json.load(f)
			self.saved_data = full_config.get(self.step_name, {})
		return self.saved_data
	
	def is_multi_label(self):
		return bool(self.data.get("multi_label", 0))

	def get_type(self):
		return "Multi-Label" if self.is_multi_label() else "Binary"

	def get_model(self, model_name) -> Model | None:
		model_data = self.data.get("models").get(model_name)
		if not model_data:
			print_color("No model of this name found", "warning")
			return None
		
		return Model(data=model_data, name=model_name)
	
	def save_model(self, model_name: str, flat_model_data: Dict[str, Any]):
		_ = self.data # refresh the saved_data to the lastest

		# * 1. Conversion du flat a structurer
		ngram_map = {
			'unigram': "(1, 1)",
			'bigram': "(1, 2)",
			'trigram': "(1, 3)",
			'quadrugram': "(1, 4)",
			"bigram_only": "(2, 2)",
		}
		vectorizer_keys = {"ngram_range", "min_df", "max_features", "vectorizer_type"}
		vectorizer_data = {}
		model_params = {}

		for key, value in flat_model_data.items():
			if key in vectorizer_keys:
				if key == "ngram_range":
					value = ngram_map.get(value, "(1, 1)")
				vectorizer_data[key] = value
			else:
				model_params[key] = value

		model_params["vectorizer"] = vectorizer_data

		# * 2. Introduction du nouveau model dans la structure local (Step1)
		if "models" not in self.saved_data:
			self.saved_data["models"] = {}

		self.saved_data["models"][model_name] = model_params

		# * 3. Introduction du nouveau model dans la structure global (fulljson)
		with open(self.config_path, 'r') as f:
			full_config = json.load(f)

		if self.step_name not in full_config:
			full_config[self.step_name] = {}
		full_config[self.step_name]["models"] = self.saved_data["models"]

		# * 4. On réecrit tout le fichier pour save le model
		with open(self.config_path, 'w') as f:
			json.dump(full_config, f, indent=4)
		
		print_color(f"Model ({model_name}) saved in config file", "success")

	def get_default_bow(self):
		return CountVectorizer(ngram_range = self.tf_idf.ngrams, max_features = self.tf_idf.max_features)
	
	def get_default_tf_id(self):
		return TfidfVectorizer(ngram_range = self.tf_idf.ngrams, max_features = self.tf_idf.max_features)
		
class ConfigLoader:
	config_data: Dict[str, Any] = {}
	config_path: str = "../config.json"

	@classmethod
	def load_json(cls):
		if os.path.exists(cls.config_path):
			with open(cls.config_path, 'r') as f:
				cls.config_data = json.load(f)
		else:
			raise FileNotFoundError(f"Config file not found at {cls.config_path}")
	@classmethod
	def _load_step(cls, step_name):
		cls.load_json()

		general = {
			"result_dir": cls.config_data.get("result_dir", "results"),
			"static_dir": cls.config_data.get("static_dir", "static"),
			"random_state": cls.config_data.get("random_state", 42),
			"test_ratio": cls.config_data.get("test_ratio", 0.2),
		}

		step1_data = cls.config_data.get(step_name, {})
		vec_data = step1_data.get("vectorizer", {})
		tfidf_data = vec_data.get("TF-IDF", {})
		bow_data = vec_data.get("BoW", {})


		tf_idf_config = VectorizerConfig(
			ngrams= parse_ngram_range(tfidf_data.get("ngrams", "(1,1)")),
			max_features= bow_data.get("max_features", None),

		)
		bow_config = VectorizerConfig(
			ngrams= parse_ngram_range(bow_data.get("ngrams", "(1,1)")),
			max_features= bow_data.get("max_features", None),
		)

		return StepConfig(
			config_path		= cls.config_path,
			step_name		= step_name,
			result_dir		= general["result_dir"],
			static_dir		= general["static_dir"],
			random_state	= general["random_state"],
			test_ratio		= general["test_ratio"],
			tf_idf			= tf_idf_config,
			bow				= bow_config,
		)
	
	@classmethod
	def load_step1(cls) -> StepConfig:
		return cls._load_step("Step1")

	@classmethod
	def load_step2(cls) -> StepConfig:
		return cls._load_step("Step2")

	@classmethod
	def load_step3(cls):
		return cls._load_step("Step3")
