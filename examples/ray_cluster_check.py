"""Проверка, что кластер действительно собран из нескольких машин.

Запускается рангом 0 как точка входа Ray-задачи. Смысл — не посчитать, а
доказать: кластер из N узлов и кластер из одного выглядят для кода одинаково,
пока не спросишь, где именно выполнялась работа.

Кладётся во вкладку Ray как «точка входа».
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import time

# ДО импорта ray, и нужно даже при подключении к чужому кластеру: драйвер
# заводит свой каталог сессии, а путь unix-сокета не может быть длиннее 103
# байт — каталог задачи в лимит не влезает.
os.environ.setdefault("RAY_TMPDIR", os.environ["LOOMA_TASK_TMP"])

import ray  # noqa: E402

# Адрес кластера приходит в RAY_ADDRESS: его ставит ранг 0 своей точке входа,
# и его же можно задать руками, чтобы подключиться к уже стоящему кластеру.
ray.init(log_to_driver=False)


@ray.remote
def where_am_i(n: int) -> dict:
    """Кто и где это посчитал. Имя узла — единственное честное доказательство
    того, что работа разъехалась по машинам."""
    return {"n": n * n, "host": socket.gethostname(), "node": ray.get_runtime_context().get_node_id()}


started = time.time()
answers = ray.get([where_am_i.remote(i) for i in range(200)])
seconds = time.time() - started

alive = [n for n in ray.nodes() if n.get("Alive")]
by_node: dict = {}
for answer in answers:
    by_node[answer["node"]] = by_node.get(answer["node"], 0) + 1

report = {
    "узлов в кластере": len(alive),
    "адреса узлов": sorted({n.get("NodeManagerAddress", "") for n in alive}),
    "имена машин": sorted({a["host"] for a in answers}),
    "задач посчитано": len(answers),
    "распределение по узлам": by_node,
    "сумма квадратов": sum(a["n"] for a in answers),
    "секунд": round(seconds, 2),
    "ресурсы кластера": ray.cluster_resources(),
}

# Главная строчка отчёта: работа шла больше чем на одной машине.
report["разъехалось по машинам"] = len(by_node) > 1

out = pathlib.Path(os.environ["LOOMA_TASK_OUT"]) / "cluster.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
ray.shutdown()
