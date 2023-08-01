from django.db import models

class LLMTransaction(models.Model):
    query = models.TextField(null=True, blank=True)
    response = models.TextField(null=True, blank=True)
    ip = models.CharField(max_length=50, null=False, blank=False,default='127.0.0.1')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
