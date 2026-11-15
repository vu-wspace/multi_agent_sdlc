r"""Thử module đăng nhập do pipeline sinh ra (workspace/src/auth).

cmd:        docker run --rm -i --network none -v "%cd%\workspace:/app:ro" sdlc-sandbox python - < demo_auth.py
PowerShell: Get-Content demo_auth.py | docker run --rm -i --network none -v "${PWD}\workspace:/app:ro" sdlc-sandbox python -
"""
import sys
sys.path.insert(0, "/app")  # chạy qua stdin nên phải tự thêm thư mục workspace

from src.auth import AccountLockedError, AuthService, InMemoryUserRepository, InvalidCredentialsError, PasswordHasher

auth = AuthService(InMemoryUserRepository(), PasswordHasher())
auth.register("an@example.com", "MatKhau#2026")
print("Đăng ký: an@example.com")

print("Đăng nhập đúng:", auth.login("an@example.com", "MatKhau#2026").email)

for i in range(1, 7):
    try:
        auth.login("an@example.com", "sai-mat-khau")
    except InvalidCredentialsError:
        print(f"Lần sai {i}: sai thông tin")
    except AccountLockedError:
        print(f"Lần sai {i}: TÀI KHOẢN ĐÃ BỊ KHÓA")

try:
    auth.login("an@example.com", "MatKhau#2026")
except AccountLockedError:
    print("Đăng nhập đúng sau khi bị khóa: vẫn bị chặn")

auth.unlock("an@example.com")
print("Sau unlock, đăng nhập đúng:", auth.login("an@example.com", "MatKhau#2026").email)
