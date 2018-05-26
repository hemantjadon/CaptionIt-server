import uuid

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from .logic import predict_from_storage

@csrf_exempt
def api_view(request):
  if (request.method == 'POST'):
    image = request.FILES.get('image')
    img_name_old = image.name
    img_ext = img_name_old.split('.')[1]
    img_name = uuid.uuid4().hex + '.' + img_ext
    storage_name = 'tmp/' + img_name
    path = default_storage.save(storage_name, ContentFile(image.read()))
    prediction = predict_from_storage(path)
    default_storage.delete(path)
    return JsonResponse({
      'ok': True,
      'status': 200,
      'caption': prediction
    })
  else:
    return JsonResponse({
      'ok': False,
      'status': 400,
      'reason': 'This API endpoint only accepts POST requests.'
    }, status=400)
