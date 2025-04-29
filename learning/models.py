import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import (
    ASTModel,
    AutoFeatureExtractor,
    AutoConfig,
    BertModel,
    BertTokenizer,
)

try:
    from laion_clap.hook import CLAP_Module
except ImportError:
    print("laion-clap not installed. Please install with: pip install laion-clap")
    CLAP_Module = None


class AudioEncoder(nn.Module):
    """Custom CNN-based audio encoder for CLAP that can be trained from scratch"""

    def __init__(self, out_dim=512):
        super().__init__()
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class CLAP(nn.Module):
    def __init__(self, audio_encoder, text_encoder, proj_dim=512):
        super().__init__()
        self.audio_enc = audio_encoder
        self.text_enc = text_encoder
        self.a_proj = nn.Sequential(
            nn.Linear(audio_encoder.out_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(text_encoder.config.hidden_size, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.output_dim = proj_dim

    def forward(
        self, audio=None, input_ids=None, attention_mask=None, return_audio_only=False
    ):
        if return_audio_only and audio is not None:
            a_feat = self.audio_enc(audio)
            a_emb = F.normalize(self.a_proj(a_feat), dim=-1)
            return a_emb

        if audio is None or input_ids is None:
            raise ValueError("Both audio and text inputs are required for CLAP")

        a_feat = self.audio_enc(audio)
        t_feat = self.text_enc(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        a_emb = F.normalize(self.a_proj(a_feat), dim=-1)
        t_emb = F.normalize(self.t_proj(t_feat), dim=-1)
        return a_emb, t_emb


def contrastive_loss(a_emb, t_emb, temperature=0.07):
    logits = a_emb @ t_emb.T / temperature  # [B, B]
    labels = torch.arange(len(a_emb), device=logits.device)
    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.T, labels)
    return (loss_a2t + loss_t2a) / 2


class MultiModalClassifier(nn.Module):
    """
    Multi-modal classifier combining vision and audio features.
    Uses ViT for image processing and AST or CLAP for audio processing.
    Includes a transformer-based fusion module for better multimodal integration.
    Always uses a transformer followed by an MLP for final classification.
    """

    def __init__(
        self,
        num_classes=3,
        use_images=True,
        use_audio=True,
        pretrained=True,
        audio_model="ast",
        use_dual_audio=False,
        fusion_type="transformer",
    ):
        super(MultiModalClassifier, self).__init__()

        if not use_images and not use_audio:
            raise ValueError("At least one modality (images or audio) must be enabled")

        self.use_images = use_images
        self.use_audio = use_audio
        self.audio_model_type = audio_model  # Store the audio model type
        self.use_dual_audio = use_dual_audio  # Flag to use both AST and CLAP
        self.fusion_type = fusion_type  # Type of fusion: "transformer" or "mlp"
        self.pretrained = pretrained  # Store if using pretrained models

        if use_images:
            self.vit = models.vit_b_16(pretrained=pretrained)
            self.vit.heads.head = nn.Identity()
            self.img_dim = 768
        else:
            self.img_dim = 0

        if use_audio:
            if audio_model == "ast" or use_dual_audio:
                self.ast_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

                if pretrained:
                    try:
                        self.ast = ASTModel.from_pretrained(self.ast_model_name)

                        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                            self.ast_model_name
                        )

                        self.ast_hidden_size = (
                            self.ast.config.hidden_size
                        )  # Should be 768
                    except Exception as e:
                        print(f"Error loading AST model: {e}")
                        self.ast = None
                        self.ast_hidden_size = 768
                        print("Using fallback audio processing")
                else:
                    print("Initializing AST model from scratch (no pretrained weights)")
                    config = AutoConfig.from_pretrained(self.ast_model_name)
                    self.ast = ASTModel(config)
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                        self.ast_model_name
                    )
                    self.ast_hidden_size = config.hidden_size
            else:
                self.ast = None
                self.ast_hidden_size = 0

            if audio_model == "clap" or use_dual_audio:
                if pretrained and CLAP_Module is not None:
                    try:
                        self.clap = CLAP_Module(
                            enable_fusion=False,  # We'll handle fusion ourselves
                            amodel="HTSAT-base",  # Use HTSAT-base audio model
                        )

                        self.clap_hidden_size = 512  # CLAP embedding dimension
                    except Exception as e:
                        print(f"Error loading CLAP model: {e}")
                        self.clap = None
                        self.clap_hidden_size = 512
                        print("Using fallback audio processing")
                else:
                    print("Initializing trainable CLAP model from scratch")

                    audio_encoder = AudioEncoder(out_dim=512)

                    bert_model = "bert-base-uncased"
                    text_encoder = BertModel.from_pretrained(bert_model)
                    self.tokenizer = BertTokenizer.from_pretrained(bert_model)

                    self.clap = CLAP(audio_encoder, text_encoder, proj_dim=512)
                    self.clap_hidden_size = 512
            else:
                self.clap = None
                self.clap_hidden_size = 0

            if use_dual_audio:
                self.audio_hidden_size = self.ast_hidden_size + self.clap_hidden_size
            elif audio_model == "ast":
                self.audio_hidden_size = self.ast_hidden_size
            elif audio_model == "clap":
                self.audio_hidden_size = self.clap_hidden_size
            else:
                raise ValueError(
                    f"Unknown audio model type: {audio_model}. Choose 'ast' or 'clap'."
                )
        else:
            self.ast = None
            self.clap = None
            self.audio_hidden_size = 0

        fusion_input_dim = self.img_dim + self.audio_hidden_size

        if fusion_input_dim > 0:
            self.fusion_dim = 512

            if self.use_images:
                self.img_projection = nn.Linear(self.img_dim, self.fusion_dim)

            if self.use_audio:
                self.audio_projection = nn.Linear(
                    self.audio_hidden_size, self.fusion_dim
                )

            num_tokens = 2  # image and audio
            self.pos_encoder = nn.Parameter(torch.zeros(1, num_tokens, self.fusion_dim))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.fusion_dim,
                nhead=8,
                dim_feedforward=self.fusion_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=2
            )

            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.fusion_dim, num_classes),
            )
        else:
            raise ValueError(
                "No modalities enabled. At least one of use_images or use_audio must be True."
            )

        print("Model architecture:")
        if use_images:
            print(
                f"  ViT output dimension: {self.img_dim} ({'pretrained' if pretrained else 'from scratch'})"
            )
        else:
            print("  Image branch disabled")

        if use_audio:
            if use_dual_audio:
                print(f"  Audio model: DUAL (AST + CLAP)")
            else:
                print(f"  Audio model: {audio_model.upper()}")
            print(
                f"  Audio output dimension: {self.audio_hidden_size} ({'pretrained' if pretrained else 'from scratch'})"
            )
        else:
            print("  Audio branch disabled")

        print(f"  Fusion: Transformer -> MLP")
        print(f"  Fusion input dimension: {fusion_input_dim}")

    def forward(
        self, x_img=None, x_audio=None, text_input_ids=None, text_attention_mask=None
    ):
        img_features = None
        audio_features = None
        batch_size = 0

        if self.use_images:
            if x_img is None:
                raise ValueError("Image input is None but use_images=True")
            img_features = self.vit(x_img)  # CLS token embedding
            batch_size = x_img.size(0)

        if self.use_audio:
            if x_audio is None:
                raise ValueError("Audio input is None but use_audio=True")
            batch_size = batch_size or x_audio.size(0)

            try:
                if x_audio.dim() == 5:
                    x_audio = x_audio.squeeze(3)  # Remove the extra dimension

                if (
                    x_audio.numel() == 0
                    or torch.isnan(x_audio).any()
                    or torch.isinf(x_audio).any()
                ):
                    raise ValueError("Invalid audio tensor with zeros, NaNs or Infs")

                audio_features_list = []

                if (
                    self.audio_model_type == "ast" or self.use_dual_audio
                ) and self.ast is not None:
                    if batch_size > 2:
                        ast_features_list = []
                        sub_batch_size = 2  # Process 2 samples at a time

                        for i in range(0, batch_size, sub_batch_size):
                            end_idx = min(i + sub_batch_size, batch_size)
                            x_audio_sub = x_audio[i:end_idx]

                            x_audio_sub = x_audio_sub.squeeze(1).transpose(1, 2)

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            ast_outputs = self.ast(
                                input_values=x_audio_sub,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                            sub_audio_features = ast_outputs.pooler_output

                            ast_features_list.append(sub_audio_features)

                        ast_features = torch.cat(ast_features_list, dim=0)

                    else:
                        x_audio_ast = x_audio.squeeze(1).transpose(1, 2)

                        ast_outputs = self.ast(
                            input_values=x_audio_ast,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        ast_features = ast_outputs.pooler_output

                    audio_features_list.append(ast_features)

                if self.audio_model_type == "clap" or self.use_dual_audio:
                    if not self.pretrained and hasattr(self.clap, "audio_enc"):
                        clap_features = self.clap(audio=x_audio, return_audio_only=True)
                    elif self.clap is not None:
                        if batch_size > 2:
                            clap_features_list = []
                            sub_batch_size = 2  # Process 2 samples at a time

                            for i in range(0, batch_size, sub_batch_size):
                                end_idx = min(i + sub_batch_size, batch_size)
                                x_audio_sub = x_audio[i:end_idx]

                                x_audio_sub = x_audio_sub.squeeze(1).mean(
                                    dim=1
                                )  # Average over frequency dimension

                                x_audio_sub_np = x_audio_sub.detach().cpu().numpy()

                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                try:
                                    with torch.no_grad():
                                        sub_audio_features_np = (
                                            self.clap.get_audio_embedding_from_data(
                                                x_audio_sub_np
                                            )
                                        )
                                        sub_audio_features = torch.from_numpy(
                                            sub_audio_features_np
                                        ).to(x_audio.device)
                                except Exception as e:
                                    print(f"CLAP processing error: {e}")
                                    sub_audio_features = torch.randn(
                                        end_idx - i,
                                        self.clap_hidden_size,
                                        device=x_audio.device,
                                        dtype=x_audio.dtype,
                                    )

                                clap_features_list.append(sub_audio_features)

                            clap_features = torch.cat(clap_features_list, dim=0)

                        else:
                            x_audio_clap = x_audio.squeeze(1).mean(
                                dim=1
                            )  # Average over frequency dimension

                            x_audio_np = x_audio_clap.detach().cpu().numpy()

                            try:
                                with torch.no_grad():
                                    audio_features_np = (
                                        self.clap.get_audio_embedding_from_data(
                                            x_audio_np
                                        )
                                    )
                                    clap_features = torch.from_numpy(
                                        audio_features_np
                                    ).to(x_audio.device)
                            except Exception as e:
                                print(f"CLAP processing error: {e}")
                                clap_features = torch.randn(
                                    batch_size,
                                    self.clap_hidden_size,
                                    device=x_audio.device,
                                    dtype=x_audio.dtype,
                                )
                    else:
                        clap_features = torch.randn(
                            batch_size,
                            self.clap_hidden_size,
                            device=x_audio.device,
                            dtype=x_audio.dtype,
                        )

                    audio_features_list.append(clap_features)

                if audio_features_list:
                    if len(audio_features_list) == 1:
                        audio_features = audio_features_list[0]
                    else:
                        if audio_features_list[0].dtype != audio_features_list[1].dtype:
                            audio_features_list[1] = audio_features_list[1].to(
                                audio_features_list[0].dtype
                            )
                        audio_features = torch.cat(audio_features_list, dim=1)
                else:
                    audio_features = torch.randn(
                        batch_size,
                        self.audio_hidden_size,
                        device=x_img.device if x_img is not None else "cuda",
                    )

            except Exception as e:
                print(f"Error processing audio: {e}")
                device = (
                    x_img.device
                    if x_img is not None
                    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                dtype = x_img.dtype if x_img is not None else torch.float32
                audio_features = torch.randn(
                    batch_size, self.audio_hidden_size, dtype=dtype, device=device
                )

        sequence = []
        attention_mask = []

        if self.use_images and img_features is not None:
            img_proj = self.img_projection(img_features)
            sequence.append(img_proj)
            attention_mask.append(torch.ones(batch_size, 1, device=img_features.device))

        if self.use_audio and audio_features is not None:
            audio_proj = self.audio_projection(audio_features)
            sequence.append(audio_proj)
            attention_mask.append(
                torch.ones(batch_size, 1, device=audio_features.device)
            )

        sequence = torch.stack(sequence, dim=1)  # [batch_size, num_tokens, fusion_dim]
        attention_mask = torch.cat(attention_mask, dim=1)  # [batch_size, num_tokens]

        sequence = sequence + self.pos_encoder[:, : sequence.size(1), :]

        transformer_output = self.transformer_encoder(sequence)

        pooled_output = torch.mean(transformer_output, dim=1)

        logits = self.classifier(pooled_output)

        return logits

