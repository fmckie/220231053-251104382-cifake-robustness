"""Draw the two CIFAKE architectures from their notebook definitions."""

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parent / '220231053_251104382_Group_Neural_Networks_Cifake.ipynb'
PDF_PATH = HERE / 'fig4_architectures.pdf'
PNG_PATH = HERE / 'fig4_architectures.png'

plt.rcParams.update({
  'font.size': 9,
  'axes.titlesize': 12,
  'figure.dpi': 300,
  'savefig.bbox': 'tight',
})


def source_node(notebook, kind, name):
  """Return one named class or function from the notebook source."""
  marker = f'{kind} {name}'
  cells = [cell for cell in notebook['cells']
           if cell['cell_type'] == 'code' and marker in ''.join(cell['source'])]
  if len(cells) != 1:
    raise ValueError(f'expected one notebook cell containing {marker}, found {len(cells)}')

  # Parse only the matching cell becasue unrelated notebook cells may contain IPython syntax.
  tree = ast.parse(''.join(cells[0]['source']))
  node_type = ast.ClassDef if kind == 'class' else ast.FunctionDef
  nodes = [node for node in tree.body
           if isinstance(node, node_type) and node.name == name]
  if len(nodes) != 1:
    raise ValueError(f'expected one {marker} definition, found {len(nodes)}')
  return nodes[0]


def number(node):
  """Evaluate the constant arithmetic used in layer dimensions."""
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
    return node.value
  if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
    return -number(node.operand)
  if isinstance(node, ast.BinOp):
    left, right = number(node.left), number(node.right)
    if isinstance(node.op, ast.Mult):
      return left * right
    if isinstance(node.op, ast.Add):
      return left + right
  raise ValueError(f'unsupported layer argument: {ast.dump(node)}')


def call_name(call):
  if isinstance(call.func, ast.Attribute):
    return call.func.attr
  if isinstance(call.func, ast.Name):
    return call.func.id
  raise ValueError(f'unsupported layer call: {ast.dump(call.func)}')


def assigned_calls(node, owner):
  specs = {}
  for item in ast.walk(node):
    if not isinstance(item, ast.Assign) or len(item.targets) != 1:
      continue
    target = item.targets[0]
    if not (isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == owner
            and isinstance(item.value, ast.Call)):
      continue
    specs[target.attr] = {
      'op': call_name(item.value),
      'args': [number(arg) for arg in item.value.args],
      'kwargs': {kw.arg: number(kw.value) for kw in item.value.keywords},
    }
  return specs


def require(actual, expected, label):
  if actual != expected:
    raise ValueError(f'{label} changed in the notebook: expected {expected}, got {actual}')


def printed_param_counts(notebook):
  counts = {}
  patterns = {
    'cnn': re.compile(r'baseline cnn parameters = ([\d,]+)'),
    'resnet': re.compile(r'resnet18 parameters = ([\d,]+)'),
  }
  for cell in notebook['cells']:
    for output in cell.get('outputs', []):
      text = output.get('text', '')
      text = ''.join(text) if isinstance(text, list) else text
      for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
          counts[key] = int(match.group(1).replace(',', ''))
  if set(counts) != set(patterns):
    raise ValueError('saved notebook output does not contain both parameter counts')
  return counts


def read_architectures():
  with NOTEBOOK.open(encoding='utf-8') as f:
    notebook = json.load(f)

  cnn_node = source_node(notebook, 'class', 'BaselineCNN')
  resnet_node = source_node(notebook, 'def', 'build_resnet18_cifar')
  cnn = assigned_calls(cnn_node, 'self')
  resnet = assigned_calls(resnet_node, 'model')

  require(cnn['conv1'], {'op': 'Conv2d', 'args': [3, 32],
                         'kwargs': {'kernel_size': 3, 'padding': 1}}, 'CNN conv1')
  require(cnn['conv2'], {'op': 'Conv2d', 'args': [32, 64],
                         'kwargs': {'kernel_size': 3, 'padding': 1}}, 'CNN conv2')
  require(cnn['pool'], {'op': 'MaxPool2d', 'args': [],
                        'kwargs': {'kernel_size': 2, 'stride': 2}}, 'CNN pool')
  require(cnn['fc1'], {'op': 'Linear', 'args': [4096, 128], 'kwargs': {}}, 'CNN fc1')
  require(cnn['fc2'], {'op': 'Linear', 'args': [128, 2], 'kwargs': {}}, 'CNN fc2')

  factory_calls = [item.value for item in resnet_node.body
                   if isinstance(item, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'model'
                           for target in item.targets)
                   and isinstance(item.value, ast.Call)]
  require(len(factory_calls), 1, 'ResNet-18 factory call count')
  require(call_name(factory_calls[0]), 'resnet18', 'ResNet-18 factory')
  require(resnet['conv1'], {'op': 'Conv2d', 'args': [3, 64],
                            'kwargs': {'kernel_size': 3, 'stride': 1,
                                       'padding': 1, 'bias': False}}, 'ResNet-18 stem')
  require(resnet['maxpool'], {'op': 'Identity', 'args': [], 'kwargs': {}},
          'ResNet-18 max-pool')
  require(resnet['fc'], {'op': 'Linear', 'args': [512, 2], 'kwargs': {}},
          'ResNet-18 head')

  counts = printed_param_counts(notebook)
  require(counts['cnn'], 544_066, 'CNN parameter count')
  require(counts['resnet'], 11_169_858, 'ResNet-18 parameter count')
  return counts


def add_block(ax, centre_x, centre_y, width, height, text, color, face_alpha=0.09):
  box = FancyBboxPatch(
    (centre_x - width / 2, centre_y - height / 2), width, height,
    boxstyle='round,pad=0.012,rounding_size=0.012',
    linewidth=1.2, edgecolor=color,
    facecolor=(*plt.matplotlib.colors.to_rgb(color), face_alpha))
  ax.add_patch(box)
  ax.text(centre_x, centre_y, text, ha='center', va='center', fontsize=8)


def draw_pipeline(ax, blocks, shapes, color, param_count, layout):
  centre_x, width, height, top, bottom, shape_x = layout
  ys = [top - i * (top - bottom) / (len(blocks) - 1) for i in range(len(blocks))]
  for index, (label, y) in enumerate(zip(blocks, ys)):
    add_block(ax, centre_x, y, width, height, label, color,
              face_alpha=0.04 if index in (0, len(blocks) - 1) else 0.10)
    if index == len(blocks) - 1:
      continue
    next_y = ys[index + 1]
    ax.annotate('', xy=(centre_x, next_y + height / 2 + 0.006),
                xytext=(centre_x, y - height / 2 - 0.006),
                arrowprops={'arrowstyle': '-|>', 'color': '0.35', 'lw': 0.9,
                            'mutation_scale': 8})
    ax.text(shape_x, (y + next_y) / 2, shapes[index], ha='left', va='center',
            fontsize=6.8, color='0.25',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.4})

  ax.text(centre_x, 0.965, f'{param_count:,} parameters', ha='center', va='center',
          color=color, fontsize=9, fontweight='bold')
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis('off')
  return ys


def draw_figure(counts):
  cnn_blocks = [
    'Input image',
    'Conv 3->32, 3x3\nReLU, max-pool 2x2',
    'Conv 32->64, 3x3\nReLU, max-pool 2x2',
    'Flatten',
    'FC 4096->128, ReLU',
    'FC 128->2',
    'REAL / FAKE logits',
  ]
  cnn_shapes = ['3 x 32 x 32', '32 x 16 x 16', '64 x 8 x 8', '4096', '128', '2']

  resnet_blocks = [
    'Input image',
    'CIFAR stem\nConv 3->64, 3x3, stride 1\nBN, ReLU; no max-pool',
    'Stage 1\n2 basic blocks, 64 channels',
    'Stage 2\n2 basic blocks, 128 channels',
    'Stage 3\n2 basic blocks, 256 channels',
    'Stage 4\n2 basic blocks, 512 channels',
    'Adaptive average pool',
    'FC 512->2',
    'REAL / FAKE logits',
  ]
  resnet_shapes = [
    '3 x 32 x 32', '64 x 32 x 32', '64 x 32 x 32', '128 x 16 x 16',
    '256 x 8 x 8', '512 x 4 x 4', '512', '2',
  ]

  # fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))  # first pass cramped the ResNet stages
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
  fig.subplots_adjust(wspace=0.06, top=0.88, left=0.035, right=0.98, bottom=0.035)
  axes[0].set_title('BaselineCNN', color='tab:blue', fontweight='bold', pad=22)
  axes[1].set_title('Adapted ResNet-18', color='tab:red', fontweight='bold', pad=22)

  draw_pipeline(axes[0], cnn_blocks, cnn_shapes, 'tab:blue', counts['cnn'],
                (0.45, 0.58, 0.075, 0.86, 0.08, 0.76))
  resnet_ys = draw_pipeline(
    axes[1], resnet_blocks, resnet_shapes, 'tab:red', counts['resnet'],
    (0.34, 0.49, 0.064, 0.86, 0.08, 0.615))

  axes[1].annotate('STEM CHANGE\nstock: 7x7, stride 2\nadapted: 3x3, stride 1\nmax-pool removed',
                   xy=(0.59, resnet_ys[1]), xytext=(0.75, resnet_ys[1]),
                   ha='left', va='center', fontsize=6.8, color='tab:red',
                   bbox={'boxstyle': 'round,pad=0.35', 'facecolor': '#fff4f2',
                         'edgecolor': 'tab:red', 'linewidth': 0.8},
                   arrowprops={'arrowstyle': '->', 'color': 'tab:red', 'lw': 0.8})
  axes[1].annotate('HEAD CHANGE\nstock: 1,000 classes\nadapted: 2 classes',
                   xy=(0.59, resnet_ys[7]), xytext=(0.75, resnet_ys[7]),
                   ha='left', va='center', fontsize=6.8, color='tab:red',
                   bbox={'boxstyle': 'round,pad=0.35', 'facecolor': '#fff4f2',
                         'edgecolor': 'tab:red', 'linewidth': 0.8},
                   arrowprops={'arrowstyle': '->', 'color': 'tab:red', 'lw': 0.8})

  fig.suptitle('CIFAKE model architectures', fontsize=14, fontweight='bold', y=0.985)
  fig.savefig(PDF_PATH, facecolor='white')
  fig.savefig(PNG_PATH, dpi=300, facecolor='white')
  plt.close(fig)


if __name__ == '__main__':
  architecture_counts = read_architectures()
  draw_figure(architecture_counts)
  for path in (PDF_PATH, PNG_PATH):
    if not path.exists() or path.stat().st_size == 0:
      raise RuntimeError(f'missing output: {path}')
    print(f'{path.name}  {path.stat().st_size} bytes')
