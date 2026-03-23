# import marimo

# __generated_with = "0.19.5"
# app = marimo.App()


# @app.cell
# def _():
def main():
    # For multi-objective studies
    import matplotlib.pyplot as plt
    import optuna

    from optuna.storages import JournalStorage, JournalFileStorage

    from optuna.visualization import plot_pareto_front, plot_param_importances
    # storage = JournalStorage(JournalFileStorage("optuna_new.log"))
    storage = JournalStorage(JournalFileStorage("optuna_flowmatching.log"))
    studies = optuna.get_all_study_summaries(storage=storage)
    print("Available studies:")
    for study in studies:
        print(f"- {study.study_name}")
    study = optuna.load_study(
        # study_name="study_CompositionalDiffusionModel",
        # study_name='no-name-a7b6b064-97b5-4d2f-be07-8ba085c972e6',
        study_name='study_flowmatching',
        storage=storage,
    )
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()

    from optuna.trial import TrialState

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of failed trials: {len(failed_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    # Retrieve the best trial
    best_trial = study.best_trial

    print(f"Best objective value: {best_trial.value}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    
    fig_history = optuna.visualization.plot_optimization_history(study)
    # fig_history.write_html("optuna_history.html")
    
    fig_importances = optuna.visualization.plot_param_importances(study)
    # fig_importances.write_html("optuna_importances.html")
    
    fig_slice = optuna.visualization.plot_slice(study)
    # fig_slice.write_html("optuna_slice.html")

# if __name__ == "__main__":
#     app.run()
if __name__ == "__main__":
    main()
