import os
import statistics

from sklearn.metrics import f1_score, precision_score, recall_score


def main(root, result_file="results.txt", label_file="version.txt", n=4 * 6):
    subs = sorted(
        (d for d in os.listdir(root) if os.path.isdir(p := os.path.join(root, d))),
        reverse=True,
    )[:n]

    # print(subs)

    y_pred = []
    y_true = []
    version_strs = []

    for sub in subs:
        path = os.path.join(root, sub, result_file)
        y_pred.append([])
        y_true.append([])
        if os.path.isfile(path):
            with open(path) as f:
                values = [
                    line
                    for line in f.read().strip().split("\n")
                    if ":" in line or "|" in line
                ]

                data = [line.split("|", 2) for line in values]
                y_pred[-1] = [pred[0].strip() for pred in data]
                y_true[-1] = [pred[1].strip() for pred in data]

        version_path = os.path.join(root, sub, label_file)
        if os.path.isfile(version_path):
            with open(version_path) as fp:
                version_strs.append(fp.read().strip())

    # print(y_pred)
    # print(y_true)
    # print(version_strs)

    def format_print(version):
        prepend = ""
        if "unique" in version:
            prepend += "T"
        elif "major" in version:
            prepend += "G"
        else:
            prepend += "GT"

        if "description" in version:
            prepend += "D"

        return prepend

    print(len(subs))

    metrics = []
    vals = []
    index = 1
    for version, pred, true in zip(version_strs, y_pred, y_true):
        label_version = format_print(version)
        print(label_version)

        f1_micro = f1_score(true, pred, average="micro")
        f1_macro = f1_score(true, pred, average="macro")

        precision = precision_score(true, pred, average="macro")
        recall = recall_score(true, pred, average="macro")

        vals.append(
            {
                "label": label_version,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "precision": precision,
                "recall": recall,
            }
        )

        if index % 4 == 0:
            metrics.append(vals)
            vals = []

        index += 1

    # metrics.append(vals)
    # print(vals[-1])

    print(len(metrics))

    for group in metrics:
        print([label["label"] for label in group])

    # print(metrics)

    for group in metrics:
        precisions = [val["precision"] for val in group]
        recalls = [val["recall"] for val in group]
        f1_micros = [val["f1_micro"] for val in group]
        f1_macros = [val["f1_macro"] for val in group]

        preva = statistics.stdev(precisions)
        recva = statistics.stdev(recalls)
        micva = statistics.stdev(f1_micros)
        macva = statistics.stdev(f1_macros)

        preev = statistics.mean(precisions)
        recev = statistics.mean(recalls)
        micev = statistics.mean(f1_micros)
        macev = statistics.mean(f1_macros)

        # for val in group:
        #     print(
        #         f"{val['label']} & {val['precision']:.3f} & {val['recall']:.3f} & {val['f1_micro']:.3f} & {val['f1_macro']:.3f} \\\\"
        #     )

        # TD  & 0.570 {\color{green}($\pm$ 0.030)} & 0.586 {\color{green}($\pm$ 0.037)} & 0.558 {\color{green}($\pm$ 0.031)} & 0.577 {\color{green}($\pm$ 0.031)} \\

        label = group[0]["label"]

        print(
            f"{label} & {preev:.3f}{{\color{{green}}($\pm$ {preva:.3f})}} & {recev:.3f}{{\color{{green}}($\pm$ {recva:.3f})}} & {micev:.3f}{{\color{{green}}($\pm$ {micva:.3f})}} & {macev:.3f}{{\color{{green}}($\pm$ {macva:.3f})}} \\\\"
        )

    return 0


if __name__ == "__main__":
    main("./mangles/")
