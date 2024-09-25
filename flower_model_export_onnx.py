import torch
import torch.nn as nn

def add_softmax_and_export_onnx(model_flowers,image_size,model_name='model_flowers.onnx'):
    model_flowers.classifier.append(nn.Softmax(dim=1))

    model_flowers.to('cpu')
    model_flowers.eval()

    torch.onnx.export(model_flowers,
                        torch.randn(1, 3, image_size, image_size),
                        model_name,
                        verbose=False,
                        input_names=['actual_input'],
                        output_names=['output'],
                        export_params=True)

    print('model exported to',model_name)
