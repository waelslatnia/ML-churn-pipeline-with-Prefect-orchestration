import argparse
import sys
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model


def main():
    """
    Point d'entree principal du pipeline ML.
    Permet d'executer chaque etape via des arguments en ligne de commande (CLI).
    """
    parser = argparse.ArgumentParser(
        description="Pipeline Machine Learning - Prediction de Churn"
    )
    
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Preparer les donnees (encodage, split, scaling)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Entrainer le modele Random Forest"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluer le modele sur les donnees de test"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Sauvegarder le modele entraine"
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Charger un modele precedemment sauvegarde"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Executer tout le pipeline entier"
    )
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    model = None
    X_train = X_test = y_train = y_test = None
    
    if args.run_all:
        print("=" * 50)
        print("Execution complete du pipeline")
        print("=" * 50)
        
        X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model)
        load_model()
        
        print("\nPipeline execute avec succes !")
        return
    
    if args.prepare:
        print("Preparation des donnees...")
        X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
        print("Donnees preparees.")
    
    if args.train:
        if X_train is None:
            print("Preparation requise. Chargement des donnees...")
            X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
        print("Entrainement du modele...")
        model = train_model(X_train, y_train)
        print("Modele entraine.")
    
    if args.evaluate:
        if X_train is None:
            print("Preparation requise. Chargement des donnees...")
            X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
        if model is None:
            print("Modele non entraine. Entrainement en cours...")
            model = train_model(X_train, y_train)
        print("Evaluation du modele...")
        evaluate_model(model, X_test, y_test)
    
    if args.save:
        if model is None:
            print("Erreur : aucun modele en memoire. Lancez --train d'abord.")
            sys.exit(1)
        save_model(model)
    
    if args.load:
        load_model()


if __name__ == "__main__":
    main()
