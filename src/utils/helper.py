import spacy

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



from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import os


def parse_ngram_range(ngram_str: str) -> Tuple[int, int]:
	"""Converts a string like "(1,2)" to a tuple (1, 2)"""
	return tuple(map(int, ngram_str.strip("() ").split(",")))


@dataclass
class VectorizerConfig:
	ngrams: Optional[Tuple[int, int]] = (1,1)
	max_features: int = None


@dataclass
class Step1Config:
	result_dir: str
	random_state: int
	test_ratio: float
	tf_idf: VectorizerConfig
	bow: VectorizerConfig

	def get_bow(self):
		return CountVectorizer(ngram_range = self.tf_idf.ngrams, max_features = self.tf_idf.max_features)
	
	def get_tf_id(self):
		return TfidfVectorizer(ngram_range = self.tf_idf.ngrams, max_features = self.tf_idf.max_features)
		


class ConfigLoader:
	config_data: Dict[str, Any] = {}
	config_path: str = "../../config.json"

	@classmethod
	def load_json(cls):
		if os.path.exists(cls.config_path):
			with open(cls.config_path, 'r') as f:
				cls.config_data = json.load(f)
		else:
			raise FileNotFoundError(f"Config file not found at {cls.config_path}")

	@classmethod
	def load_step1(cls) -> Step1Config:
		cls.load_json()

		general = {
			"result_dir": cls.config_data.get("result_dir", "results"),
			"random_state": cls.config_data.get("random_state", 42),
			"test_ratio": cls.config_data.get("test_ratio", 0.2),
		}

		step1_data = cls.config_data.get("Step1", {})
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

		return Step1Config(
			result_dir		= general["result_dir"],
			random_state	= general["random_state"],
			test_ratio		= general["test_ratio"],
			tf_idf			= tf_idf_config,
			bow				= bow_config,
		)

	@staticmethod
	def load_step2():
		# Implémentation future
		pass

	@staticmethod
	def load_step3():
		# Implémentation future
		pass


# Exemple d'utilisation :
if __name__ == "__main__":
	import pprint

	try:
		step1 = ConfigLoader.load_step1()
		pprint.pprint(step1)

		tfidf = step1.vectorizer.get_tf_idf()
		print(tfidf)
	except Exception as e:
		print(f"Erreur lors du chargement de la configuration : {e}")
