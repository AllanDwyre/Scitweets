from utils.helper import print_color

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# Classifieurs
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


class ClassifierEvaluation:
	def __init__(self, X, y, keyword_extractor, config, ticklabels, verbose = True):
		self.config = config
		self.ticklabels = ticklabels
		self.verbose = verbose

		self.X = X
		self.y = y
		self.keyword_extractor = keyword_extractor

		self.classifiers = {
			'Régression Logistique': LogisticRegression(random_state=self.config.random_state, class_weight='balanced'),
			'Naive Bayes': MultinomialNB(),
			'SVM Linéaire': LinearSVC(random_state=self.config.random_state, class_weight='balanced', max_iter=6000),
			'Random Forest': RandomForestClassifier(random_state=self.config.random_state, class_weight='balanced'),
			'Decision Tree': DecisionTreeClassifier(random_state=self.config.random_state),
			'KNN': KNeighborsClassifier(),
			'xgboost' : XGBClassifier(random_state=self.config.random_state)
		}
		self.vectoriser = {
			"TF-IDF" : self.config.get_default_tf_id(),
			"Bag of words" : self.config.get_default_bow()
		}

		self.not_scaling = ['Naive Bayes',]
	
	def evaluate(self):
		all_results = []

		if self.verbose:
			n_models = len(self.classifiers) * len(self.vectoriser)
			cols = min(3, n_models)
			rows = math.ceil(n_models / cols)

			fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
			axes = axes.flatten()


		plot_idx = 0
		for name, classifier in self.classifiers.items():

			for vectoriser_name, vectoriser in self.vectoriser.items():
				if self.verbose:	
					print_color(f"Evaluation de {name} avec {vectoriser_name}", "info")

				preprocessor = ColumnTransformer([
					("text", vectoriser, "text"),
					("numeric", 
						(StandardScaler() if name not in self.not_scaling else "passthrough"),
						make_column_selector(dtype_include="number")
					)
				])

				pipeline = Pipeline([ #Imb
					("add_keywords", self.keyword_extractor),
					("preprocessing", preprocessor),
					# ("oversampler", RandomOverSampler(random_state = self.config.random_state)),
					("clf", classifier)
				])

				y_pred_cv = cross_val_predict(pipeline, self.X, self.y, cv=5)
				cm = confusion_matrix(self.y, y_pred_cv)

				if self.verbose:
					ax = axes[plot_idx]

					sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
								xticklabels= self.ticklabels, yticklabels= self.ticklabels,
								ax=ax)

					ax.set_title(f'{name} + {vectoriser_name}')
					ax.set_xlabel("Predicted Label")
					ax.set_ylabel("True Label")

				plot_idx += 1

				cv_results = cross_validate(
					pipeline,
					self.X,
					self.y,
					cv=5,
					scoring=['f1_macro']
				)

				f1_mean = cv_results['test_f1_macro'].mean()
				f1_std = cv_results['test_f1_macro'].std()

				all_results.append({
					"Classifier": name,
					"Vectorizer": vectoriser_name,
					"F1 Mean": f1_mean,
					"F1 Std": f1_std
				})
		
		if self.verbose:
			for i in range(plot_idx, len(axes)):
				fig.delaxes(axes[i])

			plt.tight_layout()
			plt.show()

		classement_df = pd.DataFrame(all_results)
		classement_df = classement_df.sort_values(by="F1 Mean", ascending=False)

		classement_df["F1 Mean"] = classement_df["F1 Mean"].apply(lambda x : float(f"{x:.3f}"))
		classement_df["F1 Std"] = classement_df["F1 Std"].apply(lambda x : float(f"{x:.3f}"))

		return all_results, classement_df
			