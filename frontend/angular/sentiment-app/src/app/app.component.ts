import {Component, ElementRef, ViewChild} from '@angular/core';
import {ImageUploadService} from './services/image-upload.service';
import {Observable, of} from 'rxjs';
import {FormBuilder, FormGroup} from "@angular/forms";


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  loadedImg;
  sentiment: any;
  probability: any;
  error: boolean;

  constructor(private formBuilder: FormBuilder, private imageService: ImageUploadService) {

  }

  title = 'sentiment-app';
  secondScreen = false;
  selectedFile = null;
  imageUpload = null;
  form: FormGroup;
  id = null
  os = null


  ngOnInit() {
    this.form = this.formBuilder.group({
      image: ['']
    });

    const os = navigator.userAgent.toString()
    if (os.includes("Android")) {
      this.os = 3
    } else if (os.includes("Ios")) {
      this.os = 2
    } else {
      this.os = 1
    }
    console.log(this.os);
  }

  onFileSelected(event) {
    if (event.target.files.length > 0) {
      this.selectedFile = event.target.files[0];
      const file = event.target.files[0];
      this.form.get('image').setValue(file);
    }

    this.id = null;
    this.sentiment = null;
    this.probability = null;

    let image = this.selectedFile

    var reader = new FileReader();

    reader.readAsDataURL(this.selectedFile);
    reader.onload = (_event) => {
      this.loadedImg = reader.result;
    }

  }

  onUpload() {

    const formData = new FormData();
    formData.append('image', this.form.get('image').value);
    this.imageUpload = this.imageService
      .uploadImage(formData,this.os)
      .subscribe(res => {
        console.log(res);
        this.selectedFile = null;
        this.id = res['id'];
        this.sentiment = res['sentiment'];
        this.probability = res['probability'];
      });

  }

}
