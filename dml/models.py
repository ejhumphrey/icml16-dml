import optimus

GRAPH_NAME = "nlse"


def param_init(nodes, skip_biases=True):
    for n in nodes:
        for k, p in n.params.items():
            if 'bias' in k and skip_biases:
                continue
            optimus.random_init(p, 0, 0.01)


def nlse_iX_c3f2_oY(n_in, n_out, size='large', verbose=False):
    """Variable length input, 5-layer model.

    Parameters
    ----------
    n_in : int
        Length of input window, in [1, 4, 8, 10, 20].

    n_out : int
        Numer of output dimensions.

    size : str
        One of ['small', 'med', 'large', 'xlarge', 'xxlarge']

    Returns
    -------
    trainer, predictor, zero_filt : optimus.Graphs
        Fully operational optimus graphs.
    """
    # Kernel shapes
    k0, k1, k2, k3 = dict(
        small=(10, 20, 40, 96),
        med=(12, 24, 48, 128),
        large=(16, 32, 64, 192),
        xlarge=(20, 40, 80, 256),
        xxlarge=(24, 48, 96, 512))[size]

    # Input dimensions
    n0, n1, n2 = {
        1: (1, 1, 1),
        4: (3, 2, 1),
        8: (5, 3, 2),
        10: (3, 3, 1),
        20: (5, 5, 1)}[n_in]

    # Pool shapes
    p0, p1, p2 = {
        1: (1, 1, 1),
        4: (1, 1, 1),
        8: (1, 1, 1),
        10: (2, 2, 1),
        12: (2, 2, 1),
        20: (2, 2, 2)}[n_in]

    # Inputs
    # ------
    x_in = optimus.Input(
        name='x_in', shape=(None, 1, n_in, 192))
    x_same = optimus.Input(
        name='x_same', shape=x_in.shape)
    x_diff = optimus.Input(
        name='x_diff', shape=x_in.shape)
    learning_rate = optimus.Input(
        name='learning_rate', shape=None)
    margin_same = optimus.Input(
        name='margin_same', shape=None)
    margin_diff = optimus.Input(
        name='margin_diff', shape=None)
    origin_penalty = optimus.Input(
        name='origin_penalty',
        shape=None)
    inputs = [x_in, x_same, x_diff, learning_rate,
              margin_same, margin_diff, origin_penalty]

    # 1.2 Create Nodes
    logscale = optimus.Log(
        name="logscale", epsilon=1.0, gain=50.0)

    layer0 = optimus.Conv3D(
        name='layer0',
        input_shape=x_in.shape,
        weight_shape=(k0, None, n0, 13),
        pool_shape=(p0, 1),
        act_type='relu')

    layer1 = optimus.Conv3D(
        name='layer1',
        input_shape=layer0.output.shape,
        weight_shape=(k1, None, n1, 11),
        pool_shape=(p1, 1),
        act_type='relu')

    layer2 = optimus.Conv3D(
        name='layer2',
        input_shape=layer1.output.shape,
        weight_shape=(k2, None, n2, 9),
        pool_shape=(p2, 1),
        act_type='relu')

    layer3 = optimus.Affine(
        name='layer3',
        input_shape=layer2.output.shape,
        output_shape=(None, k3),
        act_type='tanh')

    layer4 = optimus.Affine(
        name='layer4',
        input_shape=layer3.output.shape,
        output_shape=(None, n_out),
        act_type='linear')

    param_nodes = [layer0, layer1, layer2, layer3, layer4]

    # 1.1 Create cloned nodes
    logscale_scopy = logscale.clone("logscale_scopy")
    logscale_dcopy = logscale.clone("logscale_dcopy")
    nodes_same = [l.clone(l.name + "_scopy") for l in param_nodes]
    nodes_diff = [l.clone(l.name + "_dcopy") for l in param_nodes]

    # 1.2 Create Loss
    # ---------------
    cost_same = optimus.Euclidean(name='cost_same')
    cost_diff = optimus.Euclidean(name='cost_diff')

    # Sim terms
    criterion = optimus.ContrastiveMargin(name='contrastive_margin')
    decay = optimus.WeightDecayPenalty(name='decay')
    total_loss = optimus.Add(name='total_loss', num_inputs=2)
    loss_nodes = [cost_same, cost_diff, criterion, decay, total_loss]

    # Graph outputs
    loss = optimus.Output(name='loss')
    z_out = optimus.Output(name='z_out')

    # 2. Define Edges
    base_edges = [
        (x_in, logscale.input),
        (logscale.output, layer0.input),
        (layer0.output, layer1.input),
        (layer1.output, layer2.input),
        (layer2.output, layer3.input),
        (layer3.output, layer4.input),
        (layer4.output, z_out)]

    base_edges_same = [
        (x_same, logscale_scopy.input),
        (logscale_scopy.output, nodes_same[0].input),
        (nodes_same[0].output, nodes_same[1].input),
        (nodes_same[1].output, nodes_same[2].input),
        (nodes_same[2].output, nodes_same[3].input),
        (nodes_same[3].output, nodes_same[4].input)]

    base_edges_diff = [
        (x_diff, logscale_dcopy.input),
        (logscale_dcopy.output, nodes_diff[0].input),
        (nodes_diff[0].output, nodes_diff[1].input),
        (nodes_diff[1].output, nodes_diff[2].input),
        (nodes_diff[2].output, nodes_diff[3].input),
        (nodes_diff[3].output, nodes_diff[4].input)]

    cost_edges = [
        (param_nodes[-1].output, cost_same.input_a),
        (nodes_same[-1].output, cost_same.input_b),
        (param_nodes[-1].output, cost_diff.input_a),
        (nodes_diff[-1].output, cost_diff.input_b),
        (cost_same.output, criterion.cost_sim),
        (cost_diff.output, criterion.cost_diff),
        (margin_same, criterion.margin_sim),
        (margin_diff, criterion.margin_diff),
        (criterion.output, total_loss.input_0),
        (origin_penalty, decay.weight),
        (param_nodes[-1].output, decay.input),
        (decay.output, total_loss.input_1),
        (total_loss.output, loss)]

    trainer_edges = optimus.ConnectionManager(
        base_edges + base_edges_same + base_edges_diff + cost_edges)

    update_manager = optimus.ConnectionManager(
        list(map(lambda n: (learning_rate, n.weights), param_nodes)) +
        list(map(lambda n: (learning_rate, n.bias), param_nodes)))

    param_init(param_nodes)

    misc_nodes = [logscale, logscale_scopy, logscale_dcopy]

    trainer = optimus.Graph(
        name=GRAPH_NAME,
        inputs=inputs,
        nodes=param_nodes + nodes_same + nodes_diff + loss_nodes + misc_nodes,
        connections=trainer_edges.connections,
        outputs=[loss, z_out],
        loss=loss,
        updates=update_manager.connections,
        verbose=verbose)

    predictor = optimus.Graph(
        name=GRAPH_NAME,
        inputs=[x_in],
        nodes=param_nodes + [logscale],
        connections=optimus.ConnectionManager(base_edges).connections,
        outputs=[z_out],
        verbose=verbose)

    return trainer, predictor


MODELS = {
    'nlse': nlse_iX_c3f2_oY
}


def create(name, **kwargs):
    return MODELS.get(name)(**kwargs)
