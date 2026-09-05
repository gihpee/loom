"""Второй шанс войти в сеть.

Со стенда: агент поднялся в момент, когда точка встречи была недоступна, не
вошёл в сеть — и остался вне неё на сутки. Управляющий поток за это время
переподключался трижды, но p2p-узел создаётся один раз за жизнь процесса, и
`on_rendezvous` при повторной регистрации молча выходил. Чинилось это релизом
на весь парк вместо одной перерегистрации.
"""

from __future__ import annotations

from looma_agent.p2p import layer


class ПоддельныйУзел:
    """Столько от PeerNode, сколько трогает слой."""

    def __init__(self, peers=()) -> None:
        self.peers = list(peers)
        self.closed = False

    def start(self, on_message=None):
        from looma_agent.p2p.peer import PeerIdentity

        return PeerIdentity(peer_id="я")

    def connected_peers(self):
        return list(self.peers)

    def in_network(self):
        return bool(self.peers)

    def close(self):
        self.closed = True

    # то, что слой вешает на таблицу связей
    def send_nowait(self, *a, **kw): return True
    def warm(self, *a, **kw): return True
    def rtt_ms(self, *a, **kw): return 0.0
    def relay_rtt_ms(self, *a, **kw): return 0.0
    def visible_addrs(self): return []


def слой(monkeypatch, узел=None, собран=None):
    """Слой с подставленной сборкой узла: настоящий Lattica здесь не нужен."""
    сделан = {}

    def построить(**options):
        сделан["options"] = options
        новый = собран if собран is not None else ПоддельныйУзел(["сосед"])
        return новый

    made = layer.PeerLayer()
    monkeypatch.setattr(layer, "lattica_available", lambda: True)
    monkeypatch.setattr(layer, "_enabled", lambda: True)
    monkeypatch.setattr(layer, "PeerNode", построить)
    monkeypatch.setattr(type(made), "_report_reachability", lambda self, _a: None)
    monkeypatch.setattr(type(made), "_start_sampler", lambda self: None)
    if узел is not None:
        made.node = узел
    return made, сделан


def test_узел_вне_сети_пересобирается(monkeypatch):
    """Ровно тот случай: узел есть, соседей нет, адреса пришли снова."""
    одинокий = ПоддельныйУзел([])
    слой_, сделан = слой(monkeypatch, узел=одинокий)

    слой_.on_rendezvous(["/dns4/looma.example/tcp/47100/p2p/aaa"], [])

    assert одинокий.closed, "старый узел обязан быть закрыт"
    assert слой_.node is not одинокий, "должен появиться новый узел"
    assert сделан["options"]["bootstraps"] == ["/dns4/looma.example/tcp/47100/p2p/aaa"]


def test_работающий_узел_не_трогают(monkeypatch):
    """Пересборка рвёт живые туннели: делать её при живой сети недопустимо."""
    живой = ПоддельныйУзел(["сосед"])
    слой_, сделан = слой(monkeypatch, узел=живой)

    слой_.on_rendezvous(["/dns4/looma.example/tcp/47100/p2p/aaa"], [])

    assert not живой.closed
    assert слой_.node is живой
    assert "options" not in сделан, "узел собирать было не нужно"


def test_без_адресов_ничего_не_ломаем(monkeypatch):
    """Пустой список — не повод закрывать то, что есть."""
    одинокий = ПоддельныйУзел([])
    слой_, _ = слой(monkeypatch, узел=одинокий)

    слой_.on_rendezvous([], [])

    assert not одинокий.closed
    assert слой_.node is одинокий


def test_первый_подъём_как_прежде(monkeypatch):
    """Обычный путь не изменился: узла нет — собираем."""
    слой_, сделан = слой(monkeypatch)

    слой_.on_rendezvous(["/dns4/looma.example/tcp/47100/p2p/aaa"], ["/relay"])

    assert слой_.node is not None
    assert сделан["options"]["relay_servers"] == ["/relay"]


# ------------------------------------------------ «в сети» значит «с точкой встречи»
def сеть(bootstraps, connected):
    """PeerNode без Lattica: проверяется только правило, а не стек."""
    from looma_agent.p2p.peer import PeerNode

    node = PeerNode.__new__(PeerNode)
    node.bootstraps = list(bootstraps)
    node.connected_peers = lambda: list(connected)
    return node


def test_одно_реле_не_считается_сетью():
    """Со стенда, и это стоило дня.

    За резервацией узел идёт к реле первым делом, поэтому соединение с ним
    есть почти всегда. Считая его входом в сеть, узел без точки встречи
    выглядит здоровым — предупреждения при старте нет, — а соседей найти не
    может: адрес по peer id ищется через DHT, вход в который даёт именно
    точка встречи.
    """
    узел = сеть(["/dns4/looma.example/tcp/47100/p2p/ТОЧКА"], connected=["РЕЛЕ"])
    assert not узел.in_network()


def test_связь_с_точкой_встречи_и_есть_сеть():
    узел = сеть(["/dns4/looma.example/tcp/47100/p2p/ТОЧКА"],
                connected=["РЕЛЕ", "ТОЧКА"])
    assert узел.in_network()


def test_без_точки_встречи_вопрос_не_стоит():
    """Некуда входить — значит и жаловаться не на что."""
    assert сеть([], connected=[]).in_network()


def test_адрес_без_идентификатора_не_ломает_проверку():
    """Такой адрес набрать нельзя, но и отказывать из-за него нельзя."""
    assert сеть(["/ip4/1.2.3.4/tcp/47100"], connected=[]).in_network()


# ---------------------------------------------- то же самое видно снаружи
def test_состояние_сети_уходит_в_телеметрию(monkeypatch):
    """Без этого поля вопрос «состоит ли узел в DHT» не имел ответа нигде.

    Значок «принимает» отвечает на обратный вопрос — дозвонятся ли ДО него, —
    а предупреждение в логе видно только тому, у кого есть доступ к машине.
    Разбирательство сводилось к чтению лога того, кто пытался позвонить.
    """
    слой_, _ = слой(monkeypatch, узел=ПоддельныйУзел(["ТОЧКА"]))
    assert слой_.status().in_network is True

    слой_, _ = слой(monkeypatch, узел=ПоддельныйУзел([]))
    assert слой_.status().in_network is False


def test_без_p2p_узла_состояние_ложно(monkeypatch):
    """Узла нет — значит и в сети его нет; врать положительным ответом нельзя."""
    слой_, _ = слой(monkeypatch)
    слой_.node = None
    assert слой_.status().in_network is False
