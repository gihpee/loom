"""Задача Looma, внутри которой работает Ray. Одна нода.

Отправляется как обычная задача — агент не знает, что внутри Ray, и знать не
должен:

    POST /admin/tasks
    {
      "command": ["python", "ray_job.py"],
      "environment": {"kind": "python", "requirements": ["ray"]},
      "resources": {"gpus": 1, "vram_gb": 16},
      "inputs": {"ray_job.py": "<этот файл в base64>"}
    }

Границы кластера — границы задачи: он поднимается при `ray.init()`, живёт
внутри одного процесса на одном узле и умирает вместе с ним. Несколько узлов —
отдельная работа над транспортом, см. docs/RAY.md.
"""

from __future__ import annotations

import json
import os
import pathlib

# ДО импорта ray. Плазма-сокет ложится внутрь этого каталога, а путь unix-сокета
# не может быть длиннее 103 байт — каталог задачи в лимит не укладывается, и
# `ray.init()` падает с ошибкой про длину пути, из которой причина не следует.
# LOOMA_TASK_TMP агент даёт как раз для такого.
os.environ.setdefault("RAY_TMPDIR", os.environ["LOOMA_TASK_TMP"])

import ray  # noqa: E402

# Дашборд наружу всё равно не смотрит: у узла нет входящих портов.
ray.init(include_dashboard=False, log_to_driver=False)


# Имя латиницей не по привычке: Ray кодирует имя задачи в ASCII, и
# кириллический идентификатор роняет воркер на UnicodeEncodeError.
@ray.remote
def square(n: int) -> int:
    return n * n


answers = ray.get([square.remote(i) for i in range(100)])

# Результат — только то, что положено в LOOMA_TASK_OUT. Всё остальное, что
# задача написала у себя, считается черновиком и никуда не поедет.
out = pathlib.Path(os.environ["LOOMA_TASK_OUT"])
out.joinpath("answer.json").write_text(json.dumps({
    "сумма": sum(answers),
    "узлов в кластере": len(ray.nodes()),
    "ресурсы": ray.cluster_resources(),
}, ensure_ascii=False, indent=2))

print("готово, сумма:", sum(answers))
ray.shutdown()
