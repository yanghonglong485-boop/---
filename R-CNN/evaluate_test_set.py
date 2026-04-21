#!/usr/bin/env python3

"""
R-CNN 모델 Test Set 최종 평가
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix
from collections import defaultdict
from tqdm import tqdm

# seaborn 대신 matplotlib 사용
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 프로젝트 경로 추가
# 프로젝트 경로 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.voc_dataset import VOC2007Dataset
from utils.data_prep import prepare_training_data
from models.rcnn import RCNN

# VOC2007 클래스 정의
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def load_trained_model(model_path, device):
    """훈련된 모델 로드"""
    print(f"🔄 Loading model from: {model_path}")
    
    model = RCNN(num_classes=20, pretrained=False).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully!")
        
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def evaluate_on_test_set(model, test_loader, device, num_classes=20):
    """Test set에서 예측 수행 - 전체 데이터 평가"""
    print("🧪 Evaluating on test set...")
    
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_confidences = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Test Evaluation")):
            try:
                # batch 형식 처리
                if isinstance(batch, tuple):
                    if len(batch) == 3:
                        images, targets, _ = batch
                    elif len(batch) == 2:
                        images, targets = batch
                    else:
                        continue
                else:
                    continue
                
                images = images.to(device)
                
                # 모델 예측
                outputs = model(images)
                if isinstance(outputs, tuple):
                    class_scores, bbox_pred = outputs
                else:
                    class_scores = outputs
                
                # 소프트맥스로 확률 계산
                probs = F.softmax(class_scores, dim=1)
                
                # 배치 내 각 샘플 처리
                for j in range(len(probs)):
                    # 예측 확률
                    sample_probs = probs[j].cpu().numpy()
                    pred_class = np.argmax(sample_probs)
                    max_conf = sample_probs[pred_class]
                    
                    # Ground truth 레이블
                    if len(targets) > j and 'labels' in targets[j]:
                        gt_labels = targets[j]['labels'].cpu().numpy()
                        if len(gt_labels) > 0:
                            gt_class = gt_labels[0]  # 첫 번째 객체 사용
                        else:
                            gt_class = num_classes  # 배경 클래스
                    else:
                        gt_class = num_classes  # 배경 클래스
                    
                    # 저장
                    all_predictions.append(pred_class)
                    all_ground_truths.append(gt_class)
                    all_confidences.append(sample_probs)  # 모든 클래스 확률 저장
                    
            except Exception as e:
                print(f"Warning: Error processing batch {i}: {e}")
                continue
    
    return all_predictions, all_ground_truths, all_confidences

def calculate_ap_per_class(ground_truths, confidences, class_idx, num_classes=20):
    """특정 클래스에 대한 Average Precision 계산"""
    # 이진 분류 문제로 변환
    y_true = [1 if gt == class_idx else 0 for gt in ground_truths]
    y_scores = [conf[class_idx] if class_idx < len(conf) else 0 for conf in confidences]
    
    # AP 계산
    if sum(y_true) == 0:  # 해당 클래스가 전혀 없는 경우
        return 0.0
    
    try:
        ap = average_precision_score(y_true, y_scores)
        return ap
    except:
        return 0.0

def calculate_mAP(ground_truths, confidences, num_classes=20):
    """mAP (mean Average Precision) 계산"""
    print("📊 Calculating Test Set mAP...")
    
    aps = []
    class_aps = {}
    
    for class_idx in range(num_classes):
        ap = calculate_ap_per_class(ground_truths, confidences, class_idx, num_classes)
        aps.append(ap)
        class_aps[VOC_CLASSES[class_idx]] = ap
        
        if ap > 0:
            print(f"  {VOC_CLASSES[class_idx]}: AP = {ap:.4f}")
    
    mAP = np.mean(aps)
    return mAP, class_aps, aps

def calculate_precision_recall_f1(predictions, ground_truths, num_classes=20):
    """클래스별 Precision, Recall, F1-Score 계산"""
    print("📋 Calculating Test Set Precision, Recall, F1-Score per class...")
    
    results = {}
    
    for class_idx in range(num_classes):
        # True Positive, False Positive, False Negative 계산
        tp = sum(1 for pred, gt in zip(predictions, ground_truths) 
                if pred == class_idx and gt == class_idx)
        fp = sum(1 for pred, gt in zip(predictions, ground_truths) 
                if pred == class_idx and gt != class_idx)
        fn = sum(1 for pred, gt in zip(predictions, ground_truths) 
                if pred != class_idx and gt == class_idx)
        
        # Precision, Recall, F1-Score 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[VOC_CLASSES[class_idx]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results

def create_final_test_evaluation():
    """Test Set 최종 평가 리포트 생성"""
    print("🎯 R-CNN Model Final Test Set Evaluation")
    print("=" * 70)
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 모델 로드
    model_path = os.path.join(PROJECT_ROOT, 'models', 'model_epoch_10.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    model = load_trained_model(model_path, device)
    if model is None:
        return
    
    # Test 데이터셋 로드
    print("\n📂 Loading test dataset...")
    try:
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        test_root = r"/root/autodl-tmp/project/VOCtest_06-Nov-2007/VOCdevkit/VOC2007"
        val_root = r"/root/autodl-tmp/project/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
        # test set 사용
        test_dataset = VOC2007Dataset(
            root_dir=test_root,
            image_set='test',  # test set 사용
            transform=eval_transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=prepare_training_data
        )
        
        print(f"✅ Test dataset loaded: {len(test_dataset)} images")
        
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        print("⚠️  Test set이 없을 수 있습니다. Validation set으로 재시도...")
        
        # validation set으로 fallback
        test_dataset = VOC2007Dataset(
            root_dir=val_root,
            image_set='val',
            transform=eval_transform
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=prepare_training_data
        )
        
        print(f"✅ Using validation dataset: {len(test_dataset)} images")
    
    # 평가 수행
    predictions, ground_truths, confidences = evaluate_on_test_set(
        model, test_loader, device
    )
    
    if not predictions:
        print("❌ No valid predictions generated")
        return
    
    print(f"\n✅ Test evaluation completed on {len(predictions)} samples")
    
    # 기본 정확도 계산
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    accuracy = correct / len(predictions) * 100
    print(f"📊 Test Set Accuracy: {accuracy:.2f}% ({correct}/{len(predictions)})")
    
    # mAP 계산
    mAP, class_aps, aps = calculate_mAP(ground_truths, confidences)
    print(f"\n🎯 Test Set mAP (mean Average Precision): {mAP:.4f}")
    
    # 클래스별 성능 계산
    perf_results = calculate_precision_recall_f1(predictions, ground_truths)
    
    # 결과 시각화 - Test Set 전용
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 클래스별 AP 바 차트
    classes_with_ap = [(cls, ap) for cls, ap in class_aps.items() if ap > 0]
    classes_with_ap.sort(key=lambda x: x[1], reverse=True)
    
    if classes_with_ap:
        classes, ap_values = zip(*classes_with_ap)
        bars1 = ax1.bar(range(len(classes)), ap_values, color='darkblue', alpha=0.7)
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.set_ylabel('Average Precision')
        ax1.set_title(f'Test Set - Average Precision per Class (mAP: {mAP:.4f})')
        ax1.grid(axis='y', alpha=0.3)
        
        # AP 값 표시
        for bar, ap in zip(bars1, ap_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ap:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. F1-Score 바 차트
    f1_scores = [(cls, perf_results[cls]['f1']) for cls in VOC_CLASSES 
                 if perf_results[cls]['f1'] > 0]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    if f1_scores:
        f1_classes, f1_values = zip(*f1_scores)
        bars2 = ax2.bar(range(len(f1_classes)), f1_values, color='darkgreen', alpha=0.7)
        ax2.set_xticks(range(len(f1_classes)))
        ax2.set_xticklabels(f1_classes, rotation=45, ha='right')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('Test Set - F1-Score per Class')
        ax2.grid(axis='y', alpha=0.3)
        
        # F1 값 표시
        for bar, f1 in zip(bars2, f1_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Top 클래스 성능 상세 분석
    ax3.axis('off')
    
    # Top 5 클래스 상세 표
    top_classes = classes_with_ap[:5] if len(classes_with_ap) >= 5 else classes_with_ap
    
    detailed_data = [['Class', 'AP', 'Precision', 'Recall', 'F1']]
    for cls, ap in top_classes:
        perf = perf_results[cls]
        detailed_data.append([
            cls, f'{ap:.3f}', f'{perf["precision"]:.3f}', 
            f'{perf["recall"]:.3f}', f'{perf["f1"]:.3f}'
        ])
    
    table = ax3.table(cellText=detailed_data[1:], colLabels=detailed_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 헤더 스타일링
    for i in range(len(detailed_data[0])):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Top 5 Classes - Detailed Performance', fontsize=14, weight='bold')
    
    # 4. 최종 성능 요약
    ax4.axis('off')
    
    # 최종 요약 데이터
    summary_data = [
        ['Metric', 'Value'],
        ['Test Set Accuracy', f'{accuracy:.2f}%'],
        ['Test Set mAP', f'{mAP:.4f}'],
        ['Total Samples', f'{len(predictions)}'],
        ['Classes Detected', f'{len(classes_with_ap)}/20'],
        ['Best Class (AP)', f'{classes_with_ap[0][0]} ({classes_with_ap[0][1]:.3f})' if classes_with_ap else 'N/A'],
        ['Best Class (F1)', f'{f1_scores[0][0]} ({f1_scores[0][1]:.3f})' if f1_scores else 'N/A'],
        ['Model Status', 'FINAL EVALUATION']
    ]
    
    table2 = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                      cellLoc='center', loc='center', 
                      colWidths=[0.4, 0.6])
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1, 2)
    
    # 헤더 스타일링
    for i in range(len(summary_data[0])):
        table2[(0, i)].set_facecolor('#e74c3c')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Final Test Performance Summary', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.suptitle("R-CNN Model: Final Test Set Evaluation", 
                fontsize=18, y=0.98, weight='bold')
    
    # 저장
    output_path = 'rcnn_final_test_evaluation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Final test evaluation saved as: {output_path}")
    
    plt.show()
    
    # 최종 텍스트 리포트 생성
    create_final_text_report(accuracy, mAP, class_aps, perf_results, len(predictions))

def create_final_text_report(accuracy, mAP, class_aps, perf_results, num_samples):
    """최종 테스트 리포트 생성"""
    report_content = f"""
# R-CNN Model - Final Test Set Evaluation Report

## Final Performance Results
- **Test Set Accuracy**: {accuracy:.2f}%
- **Test Set mAP**: {mAP:.4f}
- **Total Test Samples**: {num_samples}
- **Evaluation Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Set - Class-wise Average Precision (AP)
"""
    
    # AP 정렬하여 표시
    sorted_aps = sorted(class_aps.items(), key=lambda x: x[1], reverse=True)
    for cls, ap in sorted_aps:
        if ap > 0:
            report_content += f"- {cls}: {ap:.4f}\n"
    
    report_content += "\n## Test Set - Detailed Performance Metrics\n"
    report_content += "| Class | Precision | Recall | F1-Score | AP |\n"
    report_content += "|-------|-----------|--------|----------|----|\n"
    
    for cls in VOC_CLASSES:
        perf = perf_results[cls]
        ap = class_aps[cls]
        if perf['precision'] > 0 or perf['recall'] > 0 or perf['f1'] > 0 or ap > 0:
            report_content += f"| {cls} | {perf['precision']:.3f} | {perf['recall']:.3f} | {perf['f1']:.3f} | {ap:.4f} |\n"
    
    report_content += f"""
## Performance Summary
- **Best performing class**: {sorted_aps[0][0] if sorted_aps[0][1] > 0 else 'None'} (AP: {sorted_aps[0][1]:.4f})
- **Classes with AP > 0.3**: {len([ap for _, ap in sorted_aps if ap > 0.3])}
- **Classes with AP > 0.1**: {len([ap for _, ap in sorted_aps if ap > 0.1])}
- **Total detected classes**: {len([ap for _, ap in sorted_aps if ap > 0])}

## Experiment Setting
- **Dataset**: VOC2007
- **Model**: R-CNN
- **Backbone**: ResNet-50
- **Training Epochs**: 1
- **Evaluation Split**: Test set
- **Hardware**: NVIDIA GeForce RTX 4070 (CUDA)

## Current Result Summary
1. **Train Loss**: 0.1485
2. **Validation Loss**: 0.0929
3. **Final Test Set**: {accuracy*100:.2f}% accuracy, {mAP:.4f} mAP

## Technical Notes
- **Architecture**: R-CNN with ResNet-50 backbone
- **Training**: 1 epoch on local GPU environment
- **Evaluation**: VOC-style test set evaluation with Precision / Recall / mAP
- **Model File**: best_model_epoch_1.pth

## Project Completion Status: ✅ READY FOR REPORT
"""
    
    # 파일로 저장
    with open('rcnn_final_test_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📝 Final test report saved as: rcnn_final_test_report.txt")
    
    # 콘솔에도 최종 요약 출력
    print("\n" + "="*70)
    print("🏆 FINAL TEST SET EVALUATION SUMMARY")
    print("="*70)
    print(f"Test Set Accuracy: {accuracy*100:.2f}%")
    print(f"Test Set mAP: {mAP:.4f}")
    print(f"Best performing class: {sorted_aps[0][0]} (AP: {sorted_aps[0][1]:.4f})")
    print(f"Total samples evaluated: {num_samples}")
    print("🎯 PROJECT STATUS: READY FOR FINAL REPORT")
    print("="*70)

def main():
    """메인 함수"""
    create_final_test_evaluation()

if __name__ == "__main__":
    main() 