import ultralytics
import numpy as np
import os
import matplotlib.pyplot as plt
import contextlib

def save_plot(x, y, xlabel, ylabel, title, output_path):
    plt.figure()
    plt.plot(x, y, color='blue', label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.xlim(0, 1)  
    plt.ylim(0, 1)
    plt.savefig(output_path)
    plt.close()

def save_comparative_plot(x, y1, y2, xlabel, ylabel, title, legend_labels, output_path):
    plt.figure()
    plt.plot(x, y1, color='blue', label=legend_labels[0])
    plt.plot(x, y2, color='red', label=legend_labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xlim(0, 1)  
    plt.ylim(0, 1)
    plt.savefig(output_path)
    plt.close()

def evaluate_segmentation_model(model_path, data_yaml, imgsz=640, batch_size=16, output_dir="evaluation_results"):
    """
    Evaluate a YOLO segmentation model on a given dataset.
    
    Args:
        model_path: path to the trained YOLO model (.pt file)
        data_yaml: path to the data YAML file defining the dataset
        imgsz: image size for evaluation
        batch_size: batch size for evaluation
    """
    model = ultralytics.YOLO(model_path)
    results = model.val(data=data_yaml, split="test")
    inference_times = results.speed
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.txt")

    with open(metrics_path, "w") as f:
        with contextlib.redirect_stdout(f):
            print(model.info())
    with open(metrics_path, "a") as file:
        file.write("\nInference Time Breakdown (in ms):\n")
        file.write(f"Preprocessing Time: {inference_times['preprocess']:.4f} ms\n")
        file.write(f"Inference Time: {inference_times['inference']:.4f} ms\n")
        file.write(f"Postprocessing Time: {inference_times['postprocess']:.4f} ms\n")
        file.write(f"Loss (Model Error): {inference_times['loss']:.4f}\n")
        file.write(f"Total Inference Time: {round(inference_times['preprocess']+inference_times['inference']+inference_times['postprocess']+inference_times['loss'],3)} ms\n")
        
        file.write("\nBounding Box Metrics:\n")
        file.write(f"mAP50: {round(results.box.map50,3)}\n")
        file.write(f"mAP50-95: {round(results.box.map,3)}\n")
        file.write(f"mAP75: {round(results.box.map75,3)}\n")
        

        file.write("\nSegmentation Metrics:\n")
        file.write(f"mAP50: {round(results.seg.map50,3)}\n")
        file.write(f"mAP50-95: {round(results.seg.map,3)}\n")
        file.write(f"mAP75: {round(results.seg.map75,3)}\n")




    box_precision_curve = results.box.p_curve[0]
    box_recall_curve = results.box.r_curve[0]     
    seg_precision_curve = results.seg.p_curve[0]  
    seg_recall_curve = results.seg.r_curve[0]     

    thresholds = np.linspace(0, 1, len(box_precision_curve))


    box_f1_scores = 2 * (box_precision_curve * box_recall_curve) / (box_precision_curve + box_recall_curve + 1e-9)
    seg_f1_scores = 2 * (seg_precision_curve * seg_recall_curve) / (seg_precision_curve + seg_recall_curve + 1e-9)

    
    box_precision_path = os.path.join(output_dir, "box_precision_confidence_curve.png")
    save_plot(thresholds, box_precision_curve, "Confidence", "Precision", "Box Precision-Confidence Curve", box_precision_path)

    box_recall_path = os.path.join(output_dir, "box_recall_confidence_curve.png")
    save_plot(thresholds, box_recall_curve, "Confidence", "Recall", "Box Recall-Confidence Curve", box_recall_path)

    box_pr_curve_path = os.path.join(output_dir, "box_precision_recall_curve.png")
    save_plot(box_recall_curve, box_precision_curve, "Recall", "Precision", "Box Precision-Recall Curve", box_pr_curve_path)

    box_f1_path = os.path.join(output_dir, "box_f1_score_confidence_curve.png")
    save_plot(thresholds, box_f1_scores, "Confidence", "F1-Score", "Box F1-Score Curve", box_f1_path)

    
    seg_precision_path = os.path.join(output_dir, "seg_precision_confidence_curve.png")
    save_plot(thresholds, seg_precision_curve, "Confidence", "Precision", "Segmentation Precision-Confidence Curve", seg_precision_path)

    seg_recall_path = os.path.join(output_dir, "seg_recall_confidence_curve.png")
    save_plot(thresholds, seg_recall_curve, "Confidence", "Recall", "Segmentation Recall-Confidence Curve", seg_recall_path)

    seg_pr_curve_path = os.path.join(output_dir, "seg_precision_recall_curve.png")
    save_plot(seg_recall_curve, seg_precision_curve, "Recall", "Precision", "Segmentation Precision-Recall Curve", seg_pr_curve_path)

    seg_f1_path = os.path.join(output_dir, "seg_f1_score_confidence_curve.png")
    save_plot(thresholds, seg_f1_scores, "Confidence", "F1-Score", "Segmentation F1-Score Curve", seg_f1_path)

    
    comp_precision_path = os.path.join(output_dir, "comparative_precision_curve.png")
    save_comparative_plot(thresholds, box_precision_curve, seg_precision_curve, 
                       "Confidence", "Precision", "Box vs Segmentation Precision", 
                       ["Box Precision", "Seg Precision"], comp_precision_path)

    comp_recall_path = os.path.join(output_dir, "comparative_recall_curve.png")
    save_comparative_plot(thresholds, box_recall_curve, seg_recall_curve, 
                      "Confidence", "Recall", "Box vs Segmentation Recall", 
                      ["Box Recall", "Seg Recall"], comp_recall_path)

    comp_f1_path = os.path.join(output_dir, "comparative_f1_curve.png")
    save_comparative_plot(thresholds, box_f1_scores, seg_f1_scores, 
                      "Confidence", "F1-Score", "Box vs Segmentation F1-Score", 
                      ["Box F1", "Seg F1"], comp_f1_path)
    
    # compute precision and recall at max f1 score
    box_max_f1_idx = np.argmax(box_f1_scores)
    seg_max_f1_idx = np.argmax(seg_f1_scores)

    print(f"Thresholds at Max F1-Score:\n Box: {thresholds[box_max_f1_idx]:.3f}\n Seg: {thresholds[seg_max_f1_idx]:.3f}")

    box_max_f1_precision = box_precision_curve[box_max_f1_idx]
    box_max_f1_recall = box_recall_curve[box_max_f1_idx]
    seg_max_f1_precision = seg_precision_curve[seg_max_f1_idx]
    seg_max_f1_recall = seg_recall_curve[seg_max_f1_idx]
    with open(metrics_path, "a") as file:
        file.write("\nMetrics at Max F1-Score:\n")
        file.write(f"Box Max F1-Score: {box_f1_scores[box_max_f1_idx]:.3f} at Confidence Threshold: {thresholds[box_max_f1_idx]:.3f}\n")
        file.write(f" - Precision: {box_max_f1_precision:.3f}\n")
        file.write(f" - Recall: {box_max_f1_recall:.3f}\n")
        file.write(f"Segmentation Max F1-Score: {seg_f1_scores[seg_max_f1_idx]:.3f} at Confidence Threshold: {thresholds[seg_max_f1_idx]:.3f}\n")
        file.write(f" - Precision: {seg_max_f1_precision:.3f}\n")
        file.write(f" - Recall: {seg_max_f1_recall:.3f}\n")
    print(f"All metrics and plots saved to {output_dir}")
# Example usage:
if __name__ == "__main__":
    model_path = "orientation2.pt"
    data_yaml = "data.yaml"
    evaluate_segmentation_model(model_path, data_yaml)