from modeling import DialogElectraMaskedLM, DialogElectraMultipleChoice, DialogElectraNUP
import torch


# path = "output/electra_mlm_0"
# try:
#     model = DialogElectraMaskedLM.from_pretrained(path)
#     weights = model.dialog_electra.state_dict()
#     torch.save(weights, "weights/MLM_dialog_electra.pt")
# except:
#     print(f"failed to decomposite weights for {path}")

path = "output/electra_nup_6"
try:
    model = DialogElectraNUP.from_pretrained(path)
    # print(model)
    weights1 = model.dialog_electra.state_dict()
    weights2 = model.dialog_modeling.state_dict()
    torch.save(weights1, "weights/NUP_dialog_electra_2.pt")
    torch.save(weights2, "weights/NUP_dialog_modeling_2.pt")
except:
    print(f"failed to decomposite weights for {path}")