import torch
from torch import nn

class TestHead(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classification = nn.Sequential(
            nn.Linear(768*768, 2)
        )

    def forward(self, text_embeds, image_embeds):
        cross_modal_matrix = torch.bmm(text_embeds[:,None,:].transpose(-1,-2), image_embeds[:,None,:])
        multimodal_matrix = cross_modal_matrix.flatten(1,-1)

        multimodal_embeds = self.flatten(multimodal_matrix)
        logits = self.classification(multimodal_embeds)
        return logits

class CLIPHead(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classification = nn.Sequential(
            nn.Linear(512*512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 2),
        )

    def forward(self, text_embeds, image_embeds):
        cross_modal_matrix = torch.bmm(text_embeds[:,None,:].transpose(-1,-2), image_embeds[:,None,:])
        multimodal_matrix = cross_modal_matrix.flatten(1,-1)

        multimodal_embeds = self.flatten(multimodal_matrix)
        logits = self.classification(multimodal_embeds)
        return logits

class CLIPHeadV2(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classification = nn.Sequential(
            nn.Linear(512*512, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 2),
        )

    def forward(self, text_embeds, image_embeds):
        cross_modal_matrix = torch.bmm(text_embeds[:,None,:].transpose(-1,-2), image_embeds[:,None,:])
        multimodal_matrix = cross_modal_matrix.flatten(1,-1)

        multimodal_embeds = self.flatten(multimodal_matrix)
        logits = self.classification(multimodal_embeds)
        return logits


class CLIPHeadProjection(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.flatten = nn.Flatten()
        self.classification = nn.Sequential(
            nn.Linear(512*512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 2),
        )

    def forward(self, text_embeds, image_embeds):
        text_proj = self.text_projection(text_embeds)
        image_proj = self.image_projection(image_embeds)

        cross_modal_matrix = torch.bmm(text_proj[:,None,:].transpose(-1,-2), image_proj[:,None,:])
        multimodal_matrix = cross_modal_matrix.flatten(1,-1)

        multimodal_embeds = self.flatten(multimodal_matrix)
        logits = self.classification(multimodal_embeds)
        return logits
        

class CLIPHeadLarge(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classification = nn.Sequential(
            nn.Linear(768*768, 768),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 2),
        )

    def forward(self, text_embeds, image_embeds):
        cross_modal_matrix = torch.bmm(text_embeds[:,None,:].transpose(-1,-2), image_embeds[:,None,:])
        multimodal_matrix = cross_modal_matrix.flatten(1,-1)

        multimodal_embeds = self.flatten(multimodal_matrix)
        logits = self.classification(multimodal_embeds)
        return logits

class CLIPHeadLargeV2(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classification = nn.Sequential(
            nn.Linear(768*768, 100),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(50, 2),
        )

    def forward(self, text_embeds, image_embeds):
        cross_modal_matrix = torch.bmm(text_embeds[:,None,:].transpose(-1,-2), image_embeds[:,None,:])
        multimodal_matrix = cross_modal_matrix.flatten(1,-1)

        multimodal_embeds = self.flatten(multimodal_matrix)
        logits = self.classification(multimodal_embeds)
        return logits

class CLIPHeadProjectionLarge(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 768),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(768, 768),
        )

        self.flatten = nn.Flatten()
        self.classification = nn.Sequential(
            nn.Linear(768*768, 768),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 2),
        )

    def forward(self, text_embeds, image_embeds):
        text_proj = self.text_projection(text_embeds)
        image_proj = self.image_projection(image_embeds)

        cross_modal_matrix = torch.bmm(text_proj[:,None,:].transpose(-1,-2), image_proj[:,None,:])
        multimodal_matrix = cross_modal_matrix.flatten(1,-1)

        multimodal_embeds = self.flatten(multimodal_matrix)
        logits = self.classification(multimodal_embeds)
        return logits


class BridgeTowerHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )
    
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


class BaseLineHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_stack = nn.Sequential(
            nn.BatchNorm1d(2816), # test add batchnorm, b/c image and text encoding have diff ranges
            nn.Linear(2816, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2),
        )
    
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

class ViltHead(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 2)

        )
    
    def forward(self, x):
        logits = self.linear_stack(x)
        return logits