

# --------------------------------------------------------------- диск узла
def test_диск_считается_по_тому_а_не_по_машине(tmp_path):
    """Кэши и каталоги задач лежат на одном томе, и вытеснение считает квоты
    именно от него."""
    from looma_agent.hwinfo import disk_bytes

    free, total = disk_bytes(tmp_path)
    assert 0 < free <= total


def test_свободное_место_без_резерва_под_root(tmp_path):
    """f_bavail, а не f_bfree: часть блоков задаче не отдадут, и обещать их
    хуже, чем показать меньше."""
    import os

    from looma_agent.hwinfo import disk_bytes

    stat = os.statvfs(tmp_path)
    free, _total = disk_bytes(tmp_path)
    assert free == stat.f_frsize * stat.f_bavail


def test_недоступный_путь_даёт_нули_а_не_падение():
    """Узел без одной цифры полезнее узла, который перестал отвечать."""
    from looma_agent.hwinfo import disk_bytes

    assert disk_bytes("/такого-пути-нет-и-не-будет") == (0, 0)
