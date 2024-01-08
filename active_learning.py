import torch

def select_uncertain_samples(model, dataloader, device, num_samples=100):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for imgs, _, _, _ in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            # 여기서는 예시로 예측 확률의 표준편차를 사용
            uncertainty = preds.std(dim=1).mean().item()
            uncertainties.append((uncertainty, imgs))

    # 가장 불확실성이 높은 이미지를 선택
    uncertainties.sort(reverse=True)
    return [x[1] for x in uncertainties[:num_samples]]

def evaluate_uncertainty(model, dataloader, device):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for imgs, _, paths, _ in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            uncertainty = preds.std(dim=1).mean().item()  # 불확실성 계산
            uncertainties.extend(list(zip(paths, [uncertainty] * len(paths))))

    uncertainties.sort(key=lambda x: x[1], reverse=True)  # 불확실성이 높은 순으로 정렬
    return uncertainties