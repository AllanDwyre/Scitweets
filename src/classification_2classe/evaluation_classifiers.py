from utils.result_helper import display_result
from utils.helper import ConfigLoader, print_color

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from scipy.sparse import hstack

# Classifieurs
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class ClassifierEvaluation:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.config = ConfigLoader.load_step1()

		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

		self.classifiers = {
			'Régression Logistique': LogisticRegression(
				# max_iter=config['model_params']['classifiers']['logistic_regression']['max_iter'], 
				# C=config['model_params']['classifiers']['logistic_regression']['C'], 
				# class_weight=config['model_params']['classifiers']['logistic_regression']['class_weight']
			),
			'Naive Bayes': MultinomialNB(),
			'SVM Linéaire': LinearSVC(
				# C=config['model_params']['classifiers']['svm']['C'], 
				# class_weight=config['model_params']['classifiers']['svm']['class_weight'], 
				# dual=config['model_params']['classifiers']['svm']['dual'], 
				# max_iter=config['model_params']['classifiers']['svm']['max_iter']
			),
			'Random Forest': RandomForestClassifier(
				# n_estimators=config['model_params']['classifiers']['random_forest']['n_estimators'], 
				# class_weight=config['model_params']['classifiers']['random_forest']['class_weight'], 
				# random_state=config['model_params']['random_state']
			),
			'Decision Tree': DecisionTreeClassifier(
				# class_weight=config['model_params']['classifiers']['decision_tree']['class_weight'], 
				# random_state=config['model_params']['random_state']
			),
			'KNN': KNeighborsClassifier(
				# n_neighbors=config['model_params']['classifiers']['knn']['n_neighbors']
			)
		}
		self.vectoriser = {
			"TF-IDF" : self.config.get_tf_id(),
			"Bag of words" : self.config.get_bow()
		}
	
	def evaluate(self):

		all_results = []
		results = {}

		for name, classifier in self.classifiers.items():

			for vectoriser_name, vectoriser in self.vectoriser.items():

				# print_color(f"Evaluation de {name} avec {vectoriser_name}", "info")

				X_train_vectoriser = vectoriser.fit_transform(self.X_train["text"])
				X_test_vectoriser = vectoriser.transform(self.X_test["text"])

				X_train_numeric = self.X_train.select_dtypes(include='number')
				X_test_numeric =  self.X_test.select_dtypes(include='number')
				
				X_train_combined = hstack([X_train_vectoriser, X_train_numeric])
				X_test_combined = hstack([X_test_vectoriser, X_test_numeric])

				from sklearn.model_selection import cross_validate
				cv_results = cross_validate(
					classifier, 
					X_train_combined, 
					self.y_train,
					cv=5,
					scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
				)
	
				classifier.fit(X_train_combined, self.y_train)

				y_pred = classifier.predict(X_test_combined)

				accuracy = accuracy_score(self.y_test, y_pred)
				precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='macro')

				conf_matrix = confusion_matrix(self.y_test, y_pred)
				class_report = classification_report(self.y_test, y_pred)

				result = {
					'classifier': name,
					'vectorizer': vectoriser_name,
					'accuracy': accuracy,
					'precision': precision,
					'recall': recall,
					'f1': f1,
					'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
					'cv_accuracy_std': cv_results['test_accuracy'].std(),
					'cv_precision_mean': cv_results['test_precision_macro'].mean(),
					'cv_precision_std': cv_results['test_precision_macro'].std(),
					'cv_recall_mean': cv_results['test_recall_macro'].mean(),
					'cv_recall_std': cv_results['test_recall_macro'].std(),
					'cv_f1_mean': cv_results['test_f1_macro'].mean(),
					'cv_f1_std': cv_results['test_f1_macro'].std()
				}

				all_results.append(result)

				results[f"{name}_{vectoriser_name}"] = {
					'metrics': result,
					'confusion_matrix': conf_matrix,
					'class_report': class_report
				}

		# Classement des modèles par F1-score
		import pandas as pd
		results_df = pd.DataFrame(all_results)
		sorted_results = results_df.sort_values(by='cv_f1_mean', ascending=False)
		
		print_color("\n=== CLASSEMENT DES MODÈLES (par F1-score) ===", "header")
		for i, (idx, row) in enumerate(sorted_results.iterrows(), 1):
			print_color(f"{i}. {row['classifier']} avec {row['vectorizer']}", "info")
			print(f"   F1-score: {row['cv_f1_mean']:.4f} ± {row['cv_f1_std']:.4f}")
			print(f"   Accuracy: {row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}")
		
		return results, sorted_results

				