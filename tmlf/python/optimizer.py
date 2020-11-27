def build_sgd(model, lr=0.03):
    model.init_net.constant_fill(
        [], "one", shape=[1], value=1.0,
    )
    model.init_net.constant_fill(
        [], "lr", shape=[1], value=-lr,
    )

    for param, grad in model.get_param_to_grad_map().items():
        model.weighted_sum(
            [param, "one", grad, "lr"],
            [param],
        )
