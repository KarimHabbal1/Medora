import requests

BASE = 'http://127.0.0.1:8000/api'
ADMIN_EMAIL = 'example@admin.com'
ADMIN_PASSWORD = 'admin123'

try:
    r = requests.post(f'{BASE}/auth/signin', data={'username': ADMIN_EMAIL, 'password': ADMIN_PASSWORD})
    print('signin', r.status_code, r.text)
    r.raise_for_status()
    token = r.json()['access_token']
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    u = requests.get(f'{BASE}/admin/users', headers=headers)
    print('users', u.status_code, u.text)
    u.raise_for_status()
    users = u.json()
    patients = [x for x in users if x['role'] == 'patient']
    doctors = [x for x in users if x['role'] == 'doctor']
    print('patients count', len(patients))
    print('doctors count', len(doctors))
    if not patients or not doctors:
        raise SystemExit('No patients or doctors found')

    payload = {'patient_id': patients[0]['id'], 'doctor_id': doctors[0]['id']}
    print('payload', payload)
    a = requests.post(f'{BASE}/admin/assign-doctor', headers=headers, json=payload)
    print('assign', a.status_code, a.text)
    a.raise_for_status()
except Exception as e:
    print('exception', type(e).__name__, e)