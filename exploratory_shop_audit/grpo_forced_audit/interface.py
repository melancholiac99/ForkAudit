"""Observation-only fixes; the community action accounting and reward stay native."""
from copy import deepcopy
import contextlib
import json
import sys
from urllib.parse import urlsplit

from .common import COMMUNITY, SOURCE, INTERFACE
from exploratory_shop_audit.env_worker import Engine as OriginalEngine
from exploratory_shop_audit.runtime import SYSTEM, make_input

REJECTION = '操作被拒绝：动作不合法。请使用当前页面列出的可点击字符串或合法搜索动作。'


def route_page_name(url):
    if url is None:return None
    route=urlsplit(url).path.lstrip('/').split('/')[0]
    return route if route in {'search_results','item_page','item_sub_page','done'} else ''


class Engine(OriginalEngine):
    def __init__(self, run):
        self.last_feedback = {}
        with contextlib.redirect_stdout(sys.stderr):
            super().__init__(COMMUNITY, SOURCE/'community_products.json', SOURCE/'products.sqlite3', run/'snapshots')
        assert self.server.max_steps == 35
        # Native substring matching misreads session IDs containing page labels.
        # Read the route component; keep native transition/reward code untouched.
        self.server.get_page_name = route_page_name
        renderer = self.render
        def exact_render(public):
            rendered = renderer(public)
            prefix, _ = rendered.rsplit('可点击的按钮: ', 1)
            return prefix + '可点击的按钮: ' + json.dumps(public['actions'], ensure_ascii=False)
        self.render = exact_render

    def open(self, key, goal_index, prefix=()):
        self.last_feedback.pop(key, None)
        with contextlib.redirect_stdout(sys.stderr):
            return super().open(key, goal_index, prefix)

    def close(self, key):
        self.last_feedback.pop(key, None)
        return super().close(key)

    def state(self, key):
        state = super().state(key)
        if self.last_feedback.get(key): state['observation'] = REJECTION + '\n' + state['observation']
        state['interface_version'] = INTERFACE
        return state

    def step(self, key, action):
        # The rejection is an observation, never a substitute environment step.
        self.last_feedback[key] = False
        with contextlib.redirect_stdout(sys.stderr):
            result = super().step(key, action)
        self.last_feedback[key] = not result['legal']
        result['state'] = self.state(key)
        return result
