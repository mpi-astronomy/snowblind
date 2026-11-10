"""Sphinx directive that renders a stpipe step's ``spec`` as a parameter list.

``.. step-parameters:: snowblind.StreamCleanStep`` imports the class, reads
its spec with stpipe and lists the parameters the class adds to stpipe's
own in numpydoc's style, with the inline comment of each as its description.  For a pipeline it
also lists the step slots that snowblind fills.
"""

import importlib

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from stpipe import Pipeline, Step
from validate import Validator

GENERIC = frozenset(Step.load_spec_file().scalars)
STPIPE_DOCS = "https://stpipe.readthedocs.io/"
ITEM_NAMES = {"int_list": "integers", "float_list": "floats", "string_list": "strings"}


def describe_check(check):
    """Return (type, default) as reST text for one configobj check string."""
    kind, args, kwargs, default = Validator()._parse_check(check)
    if kind == "option":
        kind_text = "one of " + ", ".join("``None``" if a == "None" else f"``{a!r}``" for a in args)
    elif kind in ITEM_NAMES:
        lo, hi = kwargs.get("min"), kwargs.get("max")
        count = f"{lo} " if lo is not None and lo == hi else ""
        kind_text = f"list of {count}{ITEM_NAMES[kind]}"
    else:
        kind_text = kind
        lo, hi = kwargs.get("min"), kwargs.get("max")
        if lo is not None and hi is not None:
            kind_text += f", {lo} to {hi}"
        elif lo is not None:
            kind_text += f", at least {lo}"
        elif hi is not None:
            kind_text += f", at most {hi}"
    if default is None:
        default_text = ""
    elif kind in ("option", "string") and default != "None":
        default_text = f"``{default!r}``"
    else:
        default_text = f"``{default}``"
    return kind_text, default_text


def import_class(path):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


class StepParameters(Directive):
    required_arguments = 1

    def run(self):
        cls = import_class(self.arguments[0])
        spec = cls.load_spec_file()
        own = [name for name in spec.scalars if name not in GENERIC]
        lines = []
        if own:
            lines.append(":Parameters:")
            for name in own:
                kind_text, default_text = describe_check(spec[name])
                classifier = kind_text + (f", default {default_text}" if default_text else "")
                comment = (spec.inline_comments.get(name) or "").lstrip("#").strip()
                lines += [f"    **{name}** : {classifier}", f"        {comment}", ""]
        else:
            lines += ["The step has no parameters of its own.", ""]
        lines += [
            "The parameters every stpipe step has (``skip``, ``save_results``,",
            "``output_dir``, ``suffix``, ``pre_hooks``, ``post_hooks`` and the",
            f"others) are described in the `stpipe documentation <{STPIPE_DOCS}>`_.",
            "",
        ]
        if issubclass(cls, Pipeline):
            lines += ["Step slots that snowblind fills:", ""]
            for slot, step in cls.step_defs.items():
                if step.__module__.startswith("snowblind"):
                    lines.append(f"- ``{slot}``: ``{step.__module__}.{step.__name__}``")
            lines.append("")
        container = nodes.Element()
        self.state.nested_parse(StringList(lines, source=self.arguments[0]), self.content_offset, container)
        return container.children


def setup(app):
    app.add_directive("step-parameters", StepParameters)
    return {"version": "1", "parallel_read_safe": True}
