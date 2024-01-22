from collections import namedtuple
import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    """
    VGG19 has a total of 19 layers. Out of them, 'conv4_2' is used for content representation,
    and 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1' are used for style representation.
    """
    def __init__(self, requires_grad=False, show_progress=False, use_relu=True):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True, progress=show_progress).features

        self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
        self.offset = 1
        self.content_feature_maps_index = 4 
        self.style_feature_maps_indices = list(range(len(self.layer_names)))
        self.style_feature_maps_indices.remove(4)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()

        for x in range(1+self.offset):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1+self.offset, 6+self.offset):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6+self.offset, 11+self.offset):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11+self.offset, 20+self.offset):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20+self.offset, 22):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 29++self.offset):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x

        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
        
        return out


def test_net():
    model = Vgg19(requires_grad=False, show_progress=True)

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names
    print(layer_names)

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    print(content_fms_index_name)
    # (4, conv4_2)
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    print(style_fms_indices_names)
    #([0, 1, 2, 3, 5], ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1'])

    content_input = torch.zeros((1,3,224,224))
    content_img_set_of_feature_maps = model(content_input)
    print(len(content_img_set_of_feature_maps))
    target_content = content_img_set_of_feature_maps[content_fms_index_name[0]]  # [1,512,28,28]
    print(target_content.shape)
    target_content_representation = content_img_set_of_feature_maps[content_fms_index_name[0]].squeeze(axis=0)  #[512,28,28]
    print(target_content_representation.shape)

    style_input = torch.zeros((1,3,224,224))
    style_img_set_of_feature_maps = model(style_input)
    target_style_representation = [ x for cnt, x in enumerate(style_img_set_of_feature_maps) if
                                   cnt in style_fms_indices_names[0]]
    # print(len(target_style_representation))  5
    # [ 1,64,224,224  ]
    # [ 1,128,112,112 ]
    # [ 1,256,56,56 ]
    # [ 1,512,28,28 ]
    # [ 1,512,14,14 ]

    def gram_matrix(x, should_normalize=True):
        '''
        Generate gram matrices of the representations of content and style images.
        '''
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        print("after view",features.shape)
        features_t = features.transpose(1, 2)
        print("after transpose", features_t.shape)
        gram = features.bmm(features_t)
        print("after bmm", gram.shape)
        if should_normalize:
            gram /= ch * h * w
        return gram

    for target in target_style_representation:
        print("-----before----")
        print(target.shape)
        out = gram_matrix(target)
        print("-----after----")
        print(out.shape)




    pass


if __name__ == "__main__":
    test_net()