import unittest
from unittest import mock

from absl.testing import absltest
from google.adk.tools import tool_context
from google.genai import types
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools.generate import generate_images
from opal_adk.tools.generate.generate_utils import gemini_generate_image
from opal_adk.types import image_types

from google.rpc import code_pb2


class GenerateImagesTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.mock_tool_context = mock.create_autospec(
        tool_context.ToolContext, instance=True
    )
    self.mock_tool_context.save_artifact = mock.AsyncMock()
    self.mock_tool_context.state = {}

  @mock.patch.object(
      gemini_generate_image,
      'gemini_generate_images',
      autospec=True,
  )
  async def test_generate_images_pro(self, mock_generate_image):
    mock_generate_image.return_value = [
        (b'image1', 'image/png'),
        (b'image2', 'image/jpeg'),
    ]

    mock_input_images = types.Part.from_bytes(
        data=b'input1', mime_type='image/png'
    )
    self.mock_tool_context.load_artifact = mock.AsyncMock(
        return_value=mock_input_images
    )

    result = await generate_images.generate_images(
        prompt='A cool cat',
        model='pro',
        aspect_ratio='16:9',
        tool_context=self.mock_tool_context,
    )

    self.assertEqual(result['status'], 'success')
    self.assertEqual(result['message'], 'Successfully generated images')
    self.assertEqual(result['metadata']['image_name'], 'output_image_1')
    self.assertEqual(result['metadata']['mime_type'], 'image/jpeg')

    self.assertEqual(self.mock_tool_context.save_artifact.call_count, 2)
    # Check first save_artifact call
    _, kwargs1 = self.mock_tool_context.save_artifact.call_args_list[0]
    self.assertEqual(kwargs1['filename'], 'output_image_0')
    self.assertEqual(kwargs1['artifact'].inline_data.data, b'image1')

    # Check second save_artifact call
    _, kwargs2 = self.mock_tool_context.save_artifact.call_args_list[1]
    self.assertEqual(kwargs2['filename'], 'output_image_1')
    self.assertEqual(kwargs2['artifact'].inline_data.data, b'image2')

    mock_generate_image.assert_called_once()
    _, kwargs = mock_generate_image.call_args
    self.assertEqual(kwargs['model_name'], 'gemini-3-pro-image-opal')
    self.assertEqual(kwargs['aspect_ratio'], image_types.AspectRatio('16:9'))
    self.assertLen(kwargs['parts'], 2)
    self.assertIn('A cool cat', kwargs['parts'][0].text)
    self.assertEqual(kwargs['parts'][1].inline_data.data, b'input1')
    self.assertIn(
        'Successfully generated and stored an image',
        self.mock_tool_context.state['image_generated'],
    )

  @mock.patch.object(
      gemini_generate_image,
      'gemini_generate_images',
      autospec=True,
  )
  async def test_generate_images_flash(self, mock_generate_image):
    mock_generate_image.return_value = [
        (b'image1', 'image/png'),
    ]

    self.mock_tool_context.load_artifact = mock.AsyncMock(return_value=None)

    result = await generate_images.generate_images(
        prompt='A cool dog',
        model='flash',
        aspect_ratio='1:1',
        tool_context=self.mock_tool_context,
    )

    self.assertEqual(result['status'], 'success')

    mock_generate_image.assert_called_once()
    _, kwargs = mock_generate_image.call_args
    self.assertEqual(kwargs['model_name'], 'gemini-2.5-flash-image')
    self.assertEqual(kwargs['aspect_ratio'], image_types.AspectRatio('1:1'))
    self.assertLen(kwargs['parts'], 1)
    self.assertIn(
        'Successfully generated and stored an image',
        self.mock_tool_context.state['image_generated'],
    )

  async def test_generate_images_invalid_model(self):
    with self.assertRaises(opal_adk_error.OpalAdkError) as error_cm:
      await generate_images.generate_images(
          prompt='A cool dog',
          model='invalid_model',
          aspect_ratio='1:1',
          tool_context=self.mock_tool_context,
      )
    self.assertEqual(error_cm.exception.error_code, code_pb2.INVALID_ARGUMENT)
    self.assertIn(
        'generate_imagesInvalid model name: invalid_model',
        error_cm.exception.status_message,
    )


if __name__ == '__main__':
  absltest.main()
