"""Idempotent compatibility patches for the pinned native dependencies."""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / '.native'
so2 = root / 'src/Sophus/sophus/so2.cpp'
text = so2.read_text()
text = text.replace('unit_complex_.real() = 1.;', 'unit_complex_.real(1.);')
text = text.replace('unit_complex_.imag() = 0.;', 'unit_complex_.imag(0.);')
so2.write_text(text)

thread_header = root / 'src/Livox-SDK/sdk_core/src/base/thread_base.h'
text = thread_header.read_text()
if '#include <memory>' not in text:
    text = text.replace('#include <thread>', '#include <thread>\n#include <memory>')
thread_header.write_text(text)

driver_cmake = root / 'ws/src/livox_ros_driver/livox_ros_driver/CMakeLists.txt'
text = driver_cmake.read_text()
text = text.replace(
    'add_executable(${PROJECT_NAME}_node\n    ""\n    )',
    'add_executable(${PROJECT_NAME}_node EXCLUDE_FROM_ALL\n    ""\n    )',
)
driver_cmake.write_text(text)
