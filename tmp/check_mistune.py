import mistune
print(mistune.__version__)
import mistune.renderers as r
print(hasattr(r, 'AstRenderer'))
print(dir(r))
