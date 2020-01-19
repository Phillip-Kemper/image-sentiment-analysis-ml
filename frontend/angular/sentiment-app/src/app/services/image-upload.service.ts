import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {catchError, map} from "rxjs/operators";


@Injectable({
  providedIn: 'root'
})
export class ImageUploadService {
  url = "http://127.0.0.1:8000/upload/images/"
  constructor(private http:HttpClient) { }

  uploadImage(img) {
    return this.http.post(this.url, img)
      .pipe(map(res => res));

  }
}
