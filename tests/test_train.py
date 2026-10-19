from namegen.train import TrainConfig, sample_names, train


def test_short_training_run_reduces_loss(dataset):
    cfg = TrainConfig(steps=100, batch_size=128, n_layer=1, n_embd=32, eval_every=50, eval_batches=2)
    model, _, history = train(cfg, dataset=dataset, samples_per_eval=2)
    assert len(history) == 2
    assert history[-1].val_loss < history[0].val_loss
    names = sample_names(model, dataset, n=5)
    assert len(names) == 5
    assert all(set(n) <= set(dataset.vocab.chars[1:]) for n in names)
