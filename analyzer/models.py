from django.db import models


class FaceAnalysis(models.Model):
    sex_choices = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Unknown', 'Unknown'),
    ]
    
    skin_color_choices = [
        ('Fair', 'Fair'),
        ('Medium', 'Medium'),
        ('Dark', 'Dark'),
        ('Unknown', 'Unknown'),
    ]
    
    eye_color_choices = [
        ('Brown', 'Brown'),
        ('Blue', 'Blue'),
        ('Green', 'Green'),
        ('Unknown', 'Unknown'),
    ]
    
    beard_choices = [
        ('Yes', 'Yes'),
        ('No', 'No'),
        ('Unknown', 'Unknown'),
    ]
    
    sex = models.CharField(max_length=10, choices=sex_choices, default='Unknown')
    skin_color = models.CharField(max_length=10, choices=skin_color_choices, default='Unknown')
    eye_color = models.CharField(max_length=10, choices=eye_color_choices, default='Unknown')
    beard = models.CharField(max_length=10, choices=beard_choices, default='Unknown')
    
    created_at = models.DateTimeField(auto_now_add=True)  
    
    def __str__(self):
        return f"Analysis for {self.sex} with {self.beard} beard"
