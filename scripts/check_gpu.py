"""Diagnostic : quel accélérateur PyTorch voit-il dans ce conteneur ? (CPU / NVIDIA CUDA / AMD ROCm)"""
from __future__ import annotations

import torch


def main() -> None:
    hip = getattr(torch.version, "hip", None)
    cuda = torch.version.cuda
    backend = "AMD ROCm (HIP)" if hip else "NVIDIA CUDA" if cuda else "CPU uniquement"
    print(f"PyTorch        : {torch.__version__}")
    print(f"Build          : {backend}")
    # NB : sous ROCm, l'API s'appelle aussi « torch.cuda » (HIP émule CUDA)
    ok = torch.cuda.is_available()
    print(f"GPU utilisable : {'OUI' if ok else 'NON'}")
    if ok:
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i} : {torch.cuda.get_device_name(i)}")
        x = torch.randn(2048, 2048, device="cuda")
        print(f"Test calcul    : OK (somme = {float((x @ x).sum()):.1f})")
    else:
        print("-> Le pipeline tournera sur CPU (fonctionne, mais plus lent pour le CNN).")
        if cuda or hip:
            print("   Build GPU présent mais aucun périphérique visible : voir docs/gpu.md (pilotes, devices, groupes).")


if __name__ == "__main__":
    main()
