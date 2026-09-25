# 📚 โครงสร้างและแผนผังไฟล์ผลการทดลอง (Research Results File Map)
### โครงการวิจัย: Hippocampal Shape Analysis for Epilepsy Detection using Deep Learning and Statistical Shape Modeling (Dataset 1 Primary Cohort)

โฟลเดอร์นี้รวบรวมผลลัพธ์ทางสถิติ, กราฟเปรียบเทียบประสิทธิภาพโมเดล Deep Learning ทั้ง 7 สถาปัตยกรรม, ภาพจำลองการเสียรูป 3 มิติ (3D Distance Mapping), และข้อมูลดิบจากการสุ่มซ้ำ Bootstrap 1,000 รอบ สำหรับรายงานและนำเสนออาจารย์ที่ปรึกษา

---

## 🗂 ภาพรวมโครงสร้างไดเรกทอรี (Directory Overview Tree)

```text
ส่งอาจาร/
│
├── 📄 README.md                                -> เอกสารแนะนำและแผนผังโครงสร้างไฟล์ทั้งหมด (ไฟล์นี้)
│
├── 📊 performance/                             -> สรุปผลตัวเลขสถิติและประสิทธิภาพโมเดลทั้งหมด (Excel Master)
│   └── Dataset_1_Model_Performance_and_Bootstrap_Master.xlsx
│
├── 📈 p-value/                                 -> เมทริกซ์ 7x7 ทดสอบความแตกต่างอย่างมีนัยสำคัญทางสถิติ
│   ├── PValue_Matrix_7x7_AUC_Left.csv          (เมทริกซ์ความนัยสำคัญ AUC ฝั่งซ้าย)
│   └── PValue_Matrix_7x7_AUC_Right.csv         (เมทริกซ์ความนัยสำคัญ AUC ฝั่งขวา)
│
├── 📉 plot/                                    -> รูปภาพกราฟผลการทดลองทั้งหมดสำหรับลงเล่มวิทยานิพนธ์/รายงาน
│   ├── Grad cam/                               -> ภาพ 3D Mesh Distance Mapping (+3.0 SD vs -3.0 SD)
│   │   ├── distance_mapping_right_sd_plus3.png
│   │   ├── distance_mapping_right_sd_minus3.png
│   │   ├── distance_mapping_left_sd_plus3.png
│   │   └── distance_mapping_left_sd_minus3.png
│   │
│   ├── pls da/                                 -> การวิเคราะห์ลดทอนมิติและจำแนกรูปทรงด้วย PLS-DA
│   │   ├── plsda_top3_parameters_boxplot_right.png (Boxplot 3 ช่อง: PLS1, PLS2, PLS3)
│   │   ├── plsda_top3_parameters_boxplot_left.png
│   │   ├── plsda_top2_parameters_boxplot_right.png (Boxplot 2 ช่อง: PLS1, PLS2)
│   │   ├── plsda_top2_parameters_boxplot_left.png
│   │   ├── plsda_comp1_vs_10_overlay_right.png     (Scatter Plot Comp 1 vs Comp 10)
│   │   ├── plsda_comp1_vs_10_overlay_left.png
│   │   ├── pls_components_comparison_right.png     (Scree Plot Variance Explained %)
│   │   └── pls_components_comparison_left.png
│   │
│   ├── ROC/                                    -> กราฟ ROC Curves (Receiver Operating Characteristic)
│   │   ├── mean_roc_curves_all_7models_right.png   (เปรียบเทียบทั้ง 7 โมเดล ฝั่งขวา)
│   │   ├── mean_roc_curves_all_7models_left.png    (เปรียบเทียบทั้ง 7 โมเดล ฝั่งซ้าย)
│   │   ├── mean_roc_curve_best_model_MobileNet_right.png (โมเดลที่ดีที่สุดฝั่งขวา + 95% CI)
│   │   └── mean_roc_curve_best_model_MLP_left.png        (โมเดลที่ดีที่สุดฝั่งซ้าย + 95% CI)
│   │
│   ├── confusion matrix/                       -> ตาราง Confusion Matrix (True Pos/Neg, False Pos/Neg)
│   │   ├── Right/                              (7 โมเดล: MLP, MobileNet, PointNet, ResNet, ResNet+AE, SqueezeNet, SVM)
│   │   └── Left/                               (7 โมเดล: MLP, MobileNet, PointNet, ResNet, ResNet+AE, SqueezeNet, SVM)
│   │
│   └── boxplot/                                -> กราฟ Boxplot เปรียบเทียบจับคู่ตัวต่อตัว 21 คู่โมเดล (Pairwise)
│       ├── right_only/                         (21 รูปสำหรับ AUC + 21 รูปสำหรับ F1-Score)
│       └── left_only/                          (21 รูปสำหรับ AUC + 21 รูปสำหรับ F1-Score)
│
└── 📁 raw data/                                -> ข้อมูลดิบตัวเลขจากการรัน Bootstrap 1,000 รอบ
    ├── Combined_Bootstrap_Results.csv          (ไฟล์รวมรอบสุ่ม 1,000 รอบของทุกโมเดล)
    ├── Right/                                  (ไฟล์ CSV รายโมเดล 7 ไฟล์: 1,000 แถว/ไฟล์)
    └── Left/                                   (ไฟล์ CSV รายโมเดล 7 ไฟล์: 1,000 แถว/ไฟล์)
```

---

## 1. รายละเอียดโฟลเดอร์ `performance/`

รวบรวมรายงานสรุปผลเชิงตัวเลขสถิติที่สำคัญที่สุดของงานวิจัย โดยจัดทำเป็นไฟล์ Excel สมบูรณ์แบบ:

- **`Dataset_1_Model_Performance_and_Bootstrap_Master.xlsx`**
  ประกอบด้วยชีทงาน 5 ชีท:
  1. **`Performance_Summary`**: ตารางสรุปภาพรวมค่าชี้วัดประสิทธิภาพทั้ง 5 ด้าน ได้แก่ **Accuracy, Sensitivity (Recall), Specificity, F1-Score, และ AUC-ROC** ของทั้ง 7 สถาปัตยกรรมโมเดล ทั้งฝั่ง Left และ Right Hippocampus
  2. **`Model_Legend`**: ตารางถอดรหัสชื่อย่อและชื่อเต็มของโมเดล (MLP, MobileNet, PointNet, ResNet-18, ResNet+AutoEncoder, SqueezeNet, SVM)
  3. **`Final_Summary_Mean_SD`**: ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน (Mean ± SD) จากการสุ่มซ้ำ Bootstrap 1,000 รอบ เพื่อยืนยันความเสถียรของโมเดล
  4. **`PValue_Summary`**: ผลการทดสอบสมมติฐานทางสถิติแบบจับคู่ (Pairwise Paired t-tests) ระหว่างทุกคู่โมเดล ($C(7,2) = 21$ คู่) เพื่อพิสูจน์ว่าโมเดลใดมีความแตกต่างอย่างมีนัยสำคัญ
  5. **`Non_Significant_Pairs`**: รายการสรุปคู่โมเดลที่ผลลัพธ์ประสิทธิภาพใกล้เคียงกันจนไม่มีความแตกต่างอย่างมีนัยสำคัญทางสถิติ ($p \ge 0.05$)

---

## 2. รายละเอียดโฟลเดอร์ `p-value/`

ไฟล์เมทริกซ์สถิติตารางไขว้ขนาด 7x7 สำหรับตอบคำถามทางวิชาการว่า *"โมเดลแต่ละตัวมีประสิทธิภาพต่างกันจริงอย่างมีนัยสำคัญหรือไม่"*:

- **`PValue_Matrix_7x7_AUC_Left.csv`**: ตารางเมทริกซ์ 7x7 ของฮิปโปแคมปัสฝั่งซ้าย แสดงค่า $p$-value และสัญลักษณ์แสดงระดับนัยสำคัญ:
  - `***` : มีนัยสำคัญสูงยิ่ง ($p < 0.001$)
  - `**` : มีนัยสำคัญมาก ($p < 0.01$)
  - `*` : มีนัยสำคัญ ($p < 0.05$)
  - `ns` : ไม่มีความแตกต่างอย่างมีนัยสำคัญทางสถิติ (Not Significant, $p \ge 0.05$)
- **`PValue_Matrix_7x7_AUC_Right.csv`**: ตารางเมทริกซ์ 7x7 ของฮิปโปแคมปัสฝั่งขวา สำหรับค่า AUC-ROC

---

## 3. รายละเอียดโฟลเดอร์ `plot/` (กราฟและภาพประกอบสำหรับเล่มวิทยานิพนธ์)

### 3.1 โฟลเดอร์ `plot/Grad cam/` (3D Hippocampus Distance Mapping)
แสดงภาพการเสียรูปเชิงสัณฐานวิทยา (Morphological Surface Deformation) ของฮิปโปแคมปัสในพื้นที่ 3 มิติ ระหว่างกลุ่มควบคุมปกติ (Healthy Control) และกลุ่มผู้ป่วยโรคลมชัก (Epilepsy TLE) เปรียบเทียบ 2 คอมโพเนนต์หลัก (**Rank 1: PLS1** เทียบกับ **Rank 2: PLS2**) บนพื้นหลังสีขาวบริสุทธิ์ (`#FFFFFF`) พร้อมสเกลทางกายภาพแบบคงที่ (Fixed Physical Limit ในหน่วยมิลลิเมตร):

| ชื่อไฟล์ | ฝั่ง | ค่าความเบี่ยงเบน (SD Trajectory) | สเกลกายภาพ (Fixed Limit) | คำอธิบายจุดสำคัญ |
| :--- | :---: | :---: | :---: | :--- |
| `distance_mapping_right_sd_plus3.png` | Right | $+3.0\text{ SD}$ (Step 61/61) | `0.000 - 0.118 mm` | แสดงบริเวณที่มีการเคลื่อนตัวสูงสุดที่ส่วนหัวและสันหลัง (Anterior Dorsal) ของฮิปโปแคมปัสขวา |
| `distance_mapping_right_sd_minus3.png` | Right | $-3.0\text{ SD}$ (Step 1/61) | `0.000 - 0.118 mm` | แสดงโครงสร้างพื้นผิวในทิศทางการยุบตัวฝั่งตรงข้ามที่ $-3.0\text{ SD}$ |
| `distance_mapping_left_sd_plus3.png` | Left | $+3.0\text{ SD}$ (Step 61/61) | `0.000 - 0.128 mm` | แสดงบริเวณที่มีการเคลื่อนตัวสูงสุดที่ส่วนหัวของฮิปโปแคมปัสซ้าย |
| `distance_mapping_left_sd_minus3.png` | Left | $-3.0\text{ SD}$ (Step 1/61) | `0.000 - 0.128 mm` | แสดงโครงสร้างพื้นผิวฝั่งซ้ายที่ $-3.0\text{ SD}$ |

---

### 3.2 โฟลเดอร์ `plot/pls da/` (การวิเคราะห์รูปทรงด้วย Partial Least Squares Discriminant Analysis)
- **`plsda_top3_parameters_boxplot_right.png` & `left.png`**:
  - กราฟ Boxplot 3 ช่อง เรียงตามลำดับความสำคัญของตัวแปรจำแนกกลุ่ม (**Rank 1: PLS1, Rank 2: PLS2, Rank 3: PLS3**)
  - ล็อกแกน Y คงที่เท่ากันทุกตัวที่ช่วง **-80 ถึง +80**
  - แสดงเส้นคร่อมสถิติและค่า $p$-value อย่างชัดเจน (เช่น $p = 1.59 \times 10^{-111}$ ใน PLS1 ขวา)
  - ไม่แสดงจุด Outlier ให้รกตา (ตัด fliers ออกตามมาตรฐานสากล)
- **`plsda_top2_parameters_boxplot_right.png` & `left.png`**:
  - กราฟ Boxplot 2 ช่อง สำหรับกรณีที่ต้องการนำเสนอเน้นเฉพาะ **Top 2 Components (PLS1 & PLS2)**
- **`plsda_comp1_vs_10_overlay_right.png` & `left.png`**:
  - Scatter Plot เปรียบเทียบการกระจายตัวระหว่าง Component 1 (คอมโพเนนต์หลักที่แยกกลุ่มได้ชัดเจนที่สุด) และ Component 10 (จุดตัดขอบเขตตัวแปรที่เหมาะสมที่สุด)
- **`pls_components_comparison_right.png` & `left.png`**:
  - กราฟแท่งแสดง Variance Explained (%) และ Importance Score ของคอมโพเนนต์ทั้งหมด 8-10 ตัว

---

### 3.3 โฟลเดอร์ `plot/ROC/` (Receiver Operating Characteristic Curves)
- **`mean_roc_curves_all_7models_right.png` & `left.png`**:
  - กราฟ ROC เปรียบเทียบสถาปัตยกรรมโมเดล **ทั้ง 7 โมเดลในรูปเดียวกัน** (MLP, MobileNet, PointNet, ResNet, ResNet+AE, SqueezeNet, SVM)
  - แสดงค่า Mean AUC พร้อมค่าความเบี่ยงเบนมาตรฐาน $\pm\text{SD}$ ในช่อง Legend
- **`mean_roc_curve_best_model_MobileNet_right.png`**:
  - กราฟ ROC แบบละเอียดของโมเดลอันดับ 1 ฝั่งขวา (**MobileNet**) พร้อมแสดงแถบพื้นที่ช่วงความเชื่อมั่น 95% Confidence Interval
- **`mean_roc_curve_best_model_MLP_left.png`**:
  - กราฟ ROC แบบละเอียดของโมเดลอันดับ 1 ฝั่งซ้าย (**MLP**) พร้อมแถบ 95% Confidence Interval

---

### 3.4 โฟลเดอร์ `plot/confusion matrix/` (ความแม่นยำรายคลาส)
แบ่งเป็น 2 โฟลเดอร์ย่อย: `Left/` (ฝั่งซ้าย) และ `Right/` (ฝั่งขวา) รวม 14 ไฟล์:
- แต่ละไฟล์แสดงเมทริกซ์ 2x2 ระบุจำนวนตัวอย่างจริงที่ทำนาย:
  - **Healthy Control (คลาส 0)**: จำนวนที่ทายถูก (True Negative) และทายผิดเป็นลมชัก (False Positive)
  - **Epilepsy TLE (คลาส 1)**: จำนวนที่ทายถูก (True Positive) และทายหลุดเป็นปกติ (False Negative)
- สอดคล้องกับโมเดลทั้ง 7 ตัว: `MLP`, `MobileNet`, `PointNet`, `ResNet`, `ResNet_AE`, `SqueezeNet`, `SVM`

---

### 3.5 โฟลเดอร์ `plot/boxplot/` (การวิเคราะห์จับคู่เปรียบเทียบ 21 คู่)
แบ่งเป็น 2 โฟลเดอร์ย่อย: `right_only/` (ฝั่งขวา 42 ไฟล์) และ `left_only/` (ฝั่งซ้าย 42 ไฟล์):
- รวมทั้งสิ้น 84 ไฟล์ภาพ สำหรับการวิเคราะห์เปรียบเทียบแบบ Pairwise ทุกคู่ที่เป็นไปได้:
  - `pairwise_auc_pairXX_ModelA_vs_ModelB_*.png`: กราฟ Boxplot เปรียบเทียบค่า AUC ระหว่าง Model A กับ Model B พร้อมค่าสถิติ $p$-value
  - `pairwise_f1_pairXX_ModelA_vs_ModelB_*.png`: กราฟ Boxplot เปรียบเทียบค่า F1-Score ระหว่าง Model A กับ Model B

---

## 4. รายละเอียดโฟลเดอร์ `raw data/` (ข้อมูลตัวเลขดิบเพื่อความโปร่งใสและตรวจสอบซ้ำได้)

- **`Combined_Bootstrap_Results.csv`**:
  - ข้อมูลดิบรวมขนาดใหญ่ บันทึกผลลัพธ์การประเมินรอบที่ 1 ถึงรอบที่ 1,000 ของทุกโมเดล ทุกตัววัด (Accuracy, Sensitivity, Specificity, F1-Score, AUC-ROC)
- **โฟลเดอร์ย่อย `Left/` และ `Right/`**:
  - เก็บไฟล์ CSV แยกรายโมเดล 7 ไฟล์ต่อฝั่ง (รวม 14 ไฟล์ เช่น `PointNet_bootstrap_1000.csv`, `SVM_bootstrap_1000.csv` ฯลฯ)
  - แต่ละไฟล์มี 1,000 แถว เหมาะสำหรับการนำไปพล็อตใหม่ด้วยโปรแกรมสถิติภายนอก เช่น R, SPSS, หรือ GraphPad Prism

---

## 📌 สรุปผลการทดลองสำคัญเพื่อใช้ตอบคำถามอาจารย์ (Key Takeaways)

1. **สถาปัตยกรรมโมเดลที่ดีที่สุด:**
   - **Right Hippocampus (ฝั่งขวา):** **MobileNet** และ **PointNet** ให้ผลลัพธ์ประสิทธิภาพและเสถียรภาพสูงสุด
   - **Left Hippocampus (ฝั่งซ้าย):** **MLP** และ **ResNet+AutoEncoder** แสดงความสามารถในการแยกแยะสูงสุด
2. **ความสัมพันธ์เชิงสัณฐานวิทยา (Morphological Findings):**
   - ส่วนหัว (Anterior Head) และส่วน Subiculum ของฮิปโปแคมปัสเป็นบริเวณที่มีการเปลี่ยนแปลงรูปทรง (Deformation) สูงที่สุดอย่างมีนัยสำคัญในผู้ป่วยโรคลมชัก (TLE) โดยยืนยันจากทั้ง **PLS-DA Component 1 ($p < 10^{-100}$)** และการแมป 3D Distance Mapping ที่ระยะ $\pm 3.0\text{ SD}$
3. **การทดสอบความน่าเชื่อถือทางสถิติ:**
   - การสุ่มซ้ำแบบไม่ใช้พารามิเตอร์ Bootstrap 1,000 รอบ ยืนยันว่าความแตกต่างของประสิทธิภาพโมเดลไม่ได้เกิดขึ้นโดยบังเอิญ และมีระดับนัยสำคัญสถิติ $p < 0.001$ ในคู่โมเดลหลัก
