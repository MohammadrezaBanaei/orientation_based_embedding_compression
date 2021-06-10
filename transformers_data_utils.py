from transformers import AutoTokenizer, AutoModel


def get_model_weight_dict(model_name: str = "bert-base-uncased") -> dict:
    model = AutoModel.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False

    parameters_weight_dict = {
        "embedding": {
            "word_embeddings": model.embeddings.word_embeddings.weight,
            "position_embeddings": model.embeddings.position_embeddings.weight
        }
    }

    params = model.state_dict()
    number_layers = len(model.encoder.layer)

    for layer_idx in range(number_layers):
        temp_dict = {}  # dict storing parameters only for this layer
        temp_initial = "encoder.layer.%s." % layer_idx
        temp_parameters_name = [i.split(temp_initial)[1] for i in params if i.startswith(temp_initial)
                                and "bias" not in i and "LayerNorm" not in i]
        for j in temp_parameters_name:
            temp_dict[j.replace("attention.", "").replace("self.", "").replace("weight", "")[:-1].
                replace(".", "_")] = params["%s%s" % (temp_initial, j)]
        parameters_weight_dict["_".join(temp_initial.split(".")[1:-1])] = temp_dict

    return parameters_weight_dict
