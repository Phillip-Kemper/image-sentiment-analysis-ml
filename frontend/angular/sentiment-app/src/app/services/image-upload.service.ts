import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {catchError, map} from "rxjs/operators";


@Injectable({
  providedIn: 'root'
})
export class ImageUploadService {
  url = "sampleURl"
  constructor(private http:HttpClient) { }

  uploadImage(img) {
    return this.http.post(this.url, img)
      .pipe(map(res => console.log(res)));

  }
}
